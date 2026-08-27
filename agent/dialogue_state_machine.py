"""Pillar II：对话状态机 + 槽位管理（增量槽位提取 / 槽位擦除与重写）。

- 每轮把顾客消息蒸馏进 DialogueState：
    * 品类槽位（I'm looking for X）
    * 约束槽位（hard=2 / soft=1，来源回合、覆盖标记）
    * 场景信号（boundary / override / no_more_pref / vague）
- 支持突发意图覆盖：检测到 "ignore my earlier preference" 时，
  对旧 soft 槽位执行擦除（overridden=True），并用新值重写为 hard 槽位。
- 会话状态全部保存在内存（本进程内每个 session 独立，不写磁盘）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils import session_utils as su

# 消息模式（兼容官方确定性模拟器 + 预留 paraphrasing 容错）
RE_LOOKING_FOR = re.compile(r"looking for\s+([^.,;]+)", re.I)
RE_KEY_REQ = re.compile(r"key requirement is\s*[:：]?\s*(.+?)(?:\.\s*)?$", re.I)
RE_WHAT_MATTERS = re.compile(r"what matters is\s*[:：]?\s*(.+?)(?:\.\s*)?$", re.I)
RE_OVERRIDE = re.compile(r"ignore my earlier preference.*?what i need is\s*[:：]?\s*(.+?)(?:\.\s*)?$", re.I | re.S)
RE_NO_PREF_ATTR = re.compile(r"i don't have a preference for\s+([a-z_]+)", re.I)
# 模拟器实际话术是 "I don't have an additional preference for X."，
# 同时兼容 "no additional preference" 等改写（paraphrasing 容错）
RE_NO_MORE = re.compile(r"(?:no additional preference|don.t have an additional preference)", re.I)
RE_NOT_RIGHT = re.compile(r"not quite right", re.I)


@dataclass
class Constraint:
    """一条已蒸馏的约束槽位。"""

    value: str            # 原文（截断到 180 字符，与模拟器一致）
    attr_type: str        # material/color/feature/...
    tokens: tuple[str, ...]
    hardness: int         # 2=hard（必须满足） / 1=soft（倾向）
    source_turn: int
    overridden: bool = False
    is_old_pref: bool = False  # old-preference tagged at turn 1 (erased precisely on override)

    @property
    def key(self) -> str:
        return su.constraint_key(self.value)


@dataclass
class DialogueState:
    """单会话动态状态（内存态，独立于其它会话）。"""

    session_id: str
    user_profile: dict
    category_phrase: str = ""
    category_tokens: list[str] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    flags: dict = field(default_factory=dict)   # boundary_used / override_seen / no_more_pref / vague
    turn_summary: list[str] = field(default_factory=list)
    last_reply_kind: str = ""

    # ---- 便捷访问 --------------------------------------------------------
    @property
    def hard(self) -> list[Constraint]:
        return [c for c in self.constraints if c.hardness == 2 and not c.overridden]

    @property
    def soft(self) -> list[Constraint]:
        return [c for c in self.constraints if c.hardness == 1 and not c.overridden]

    @property
    def active(self) -> list[Constraint]:
        return [c for c in self.constraints if not c.overridden]

    def disclosed_values(self) -> set[str]:
        return {c.key for c in self.constraints}

    def total_constraints(self) -> int:
        return len(self.active)

    def to_query_terms(self) -> list[str]:
        """把当前状态蒸馏成检索关键词（Pillar III：上下文蒸馏产物）。"""
        terms: list[str] = []
        for c in self.active:
            terms.extend(c.tokens)
        return list(dict.fromkeys(terms))


class DialogueStateMachine:
    """状态机：new_state 建槽，update 每轮蒸馏/擦除/重写。"""

    def __init__(self, override_erase: bool = False) -> None:
        self.override_erase = override_erase

    def new_state(self, session_id: str, user_profile: dict) -> DialogueState:
        return DialogueState(session_id=session_id, user_profile=user_profile or {})

    # ------------------------------------------------------------------
    def update(self, state: DialogueState, user_message: str, turn: int) -> None:
        """把一轮顾客消息蒸馏进状态（Pillar II 槽位提取 + Pillar III 上下文蒸馏）。"""
        text = user_message or ""
        low = text.lower()

        # 0) turn 1: tag the initial-message tail as old-preference (override scenario)
        if turn == 1 and not state.turn_summary:
            m = RE_LOOKING_FOR.search(text)
            if m:
                tail = text[m.end():].strip(' .;,-\t\n')
                low_tail = tail.lower()
                if tail and not any(k in low_tail for k in ('key requirement', 'still exploring', 'what matters')):
                    self._add_constraint(state, tail[:180], hardness=1, turn=turn, old_pref=True)

        # 1) 品类槽位
        m = RE_LOOKING_FOR.search(text)
        if m:
            phrase = su.normalize(m.group(1))
            phrase = re.sub(r"\s*(but i'm|\.\s*a key requirement).*$", "", phrase)
            if phrase and phrase != state.category_phrase:
                state.category_phrase = phrase
                state.category_tokens = list(dict.fromkeys(t for t in su.tokenize(phrase) if t not in ("clothing", "shoes", "jewelry", "item")))

        # 2) 意图覆盖：槽位擦除 + 重写（Pillar II 突发覆盖）
        m = RE_OVERRIDE.search(text)
        if m:
            state.flags["override_seen"] = True
            # slot erasure (Pillar II): configurable; default conservative -> keep old
            # preference as a weak soft signal (it still describes the product and helps
            # ranking), while the new value is promoted to top-priority hard constraint.
            if self.override_erase:
                for c in state.constraints:
                    if c.is_old_pref:
                        c.overridden = True
            new_value = m.group(1).strip()
            self._add_constraint(state, new_value, hardness=2, turn=turn)

        # 3) 约束槽位提取：关键需求 / what matters（每轮最多 2 条）
        m = RE_KEY_REQ.search(text)
        if m:
            for v in su.split_values(m.group(1))[:2]:
                self._add_constraint(state, v, hardness=2, turn=turn)
        m = RE_WHAT_MATTERS.search(text)
        if m:
            for v in su.split_values(m.group(1))[:2]:
                self._add_constraint(state, v, hardness=1, turn=turn)

        # 4) 场景信号
        m = RE_NO_PREF_ATTR.search(text)
        if m:
            state.flags["boundary_used"] = True
            state.flags["boundary_attr"] = m.group(1).lower()
        if RE_NO_MORE.search(low):
            state.flags["no_more_pref"] = True
        if RE_NOT_RIGHT.search(low):
            state.flags["vague"] = True

        # 5) 通用槽位填充（paraphrasing 容错）：直接抽取材质/颜色词作 soft 槽位
        for material in ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"):
            if re.search(rf"\b{material}\b", low) and material not in state.disclosed_values():
                self._add_constraint(state, material, hardness=1, turn=turn)
        for color in ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"):
            if re.search(rf"\b{color}\b", low) and color not in state.disclosed_values():
                self._add_constraint(state, f"color: {color}", hardness=1, turn=turn)

        # 6) 蒸馏摘要（Pillar III：短时会话上下文）
        state.turn_summary.append(f"t{turn}:cat={state.category_phrase}|C={len(state.active)}")
        state.last_reply_kind = "parsed"

    # ------------------------------------------------------------------
    @staticmethod
    def _add_constraint(state: DialogueState, value: str, hardness: int, turn: int, old_pref: bool = False) -> None:
        value = (value or "").strip()
        if not value or len(value) < 2:
            return
        value = value[:180]
        key = su.constraint_key(value)
        if not key:
            return
        # 去重/升级：同键已存在 → 提升硬度并重新激活（修复 override 后新硬约束被排除的 bug）
        for c in state.constraints:
            if c.key == key:
                if hardness >= c.hardness:
                    c.hardness = hardness
                    c.overridden = False
                return
        state.constraints.append(Constraint(
            value=value,
            attr_type=su.classify_attribute(value),
            tokens=su.group_tokens(value),
            hardness=hardness,
            source_turn=turn,
            is_old_pref=old_pref,
        ))

