# TechJam2026 数据盘点 + 商品字典 + 提问价值分析报告

- 生成时间：2026-08-28 02:18:49
- 数据源：`data/catalog.jsonl`（50000 商品）+ `data/public_set.jsonl`（200 会话）

## 一、数据速览

### 字段覆盖率
| 字段 | 覆盖率 |
|---|---|
| `parent_asin` | 100.0% |
| `title` | 100.0% |
| `features` | 100.0% |
| `description` | 100.0% |
| `price` | 100.0% |
| `categories` | 100.0% |
| `details` | 100.0% |
| `average_rating` | 100.0% |
| `rating_number` | 100.0% |
| `store` | 100.0% |

### 缺失与脏数据
| 问题 | 数量 | 占比 |
|---|---|---|
| price_missing | 39473 | 78.95% |
| description_empty | 23887 | 47.77% |
| features_empty | 5219 | 10.44% |
| store_dirty | 578 | 1.16% |
| details_empty | 1670 | 3.34% |
| price_non_numeric | 117 | 0.23% |
| store_generic | 263 | 0.53% |
| title_empty | 2 | 0.00% |
| title_short_or_generic | 3 | 0.01% |
| price_zero_or_neg | 1 | 0.00% |

### 价格
- 有价格商品占比 **20.8%**；中位数 $22.88，P25 $14.99，P75 $39.99，区间 $0.0–$4119.0
  → **budget 约束必须 lenient**：79% 商品无价格，不能用 budget 硬过滤

### 品类分布（Top 二级品类）
| 二级品类 | 数量 |
|---|---|
| Women | 26406 |
| Men | 9901 |
| Novelty & More | 3376 |
| Girls | 1716 |
| Westlake | 1136 |
| Boot Shop | 1131 |
| Sport Specific Clothing | 1114 |
| Boys | 1101 |
| Baby | 1031 |
| Luggage & Travel Gear | 976 |
| Costumes & Accessories | 937 |
| Shoe, Jewelry & Watch Accessories | 436 |
| Toddler Test | 48 |
| Kids Shoes Union | 36 |
| Top 50 by Product Type | 32 |
| Uniforms, Work & Safety | 31 |
| Swimwear TEST | 29 |
| Plus-Size Fashion | 29 |
| Women's Plus-Size Apparel | 27 |
| Customers' Most-Loved: Sweaters Under $30 pASIN Test | 24 |

### 评分分布
| average_rating 区间 | 数量 |
|---|---|
| 4.0-4.5 | 17104 |
| 4.5-5.0 | 9542 |
| 5.0 | 7034 |
| 3.5-4.0 | 8881 |
| 3.0-3.5 | 4542 |
| <3.0 | 2897 |

| rating_number 区间 | 数量 |
|---|---|
| 0 | 0 |
| 1-10 | 23455 |
| 11-100 | 17224 |
| 101-1000 | 7789 |
| 1000+ | 1532 |

## 二、商品字典 vocab.json

### material（72 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| polyester | polyester(10339)、100% polyester(2998)、poly(440) |
| cotton | cotton(9414)、100% cotton(3698)、cotton blend(668) |
| fabric | fabric(7271)、soft fabric(573) |
| leather | leather(6882)、faux leather(478)、genuine leather(473) |
| rubber | rubber(6616)、rubber sole(5886)、synthetic sole(1405) |
| spandex | spandex(5412)、stretch(2825)、elastane(945) |
| denim | jeans(2509)、denim(929)、jean(432) |
| mesh | mesh(2466)、mesh fabric(113)、netting(14) |

### color（45 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| black | black(6802)、black white(90)、jet black(9) |
| white | white(3319)、off-white(14)、pure white(10) |
| silver | silver(2938)、gunmetal(40)、metallic silver(22) |
| blue | blue(2670)、light blue(161)、dark blue(100) |
| gold | gold(2444)、golden(171)、metallic gold(10) |
| pattern | printed(2192)、floral(1657)、pattern(1340) |
| red | red(1916)、crimson(26)、scarlet(23) |
| grey | grey(1741)、heather grey(599)、gray(571) |

### size（34 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| s | s(28194)、small(2942) |
| shoe_size | 5(9364)、8(4290)、10(4246) |
| tall | long(7879)、tall(623) |
| numeric | 2(7408)、4(5547)、8(4290) |
| petite | short(4223)、petite(251) |
| m | m(3733)、medium(1967)、regular(1053) |
| waist_inseam | waist(3669)、inseam(547)、waist 32(28) |
| one_size | adjustable(3542)、one size(1716)、free size(65) |

### style（44 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| romantic | soft(9250)、feminine(451)、romantic(127) |
| casual | casual(7486)、everyday(1879)、weekend(362) |
| trendy | fashion(5006)、stylish(2769)、trendy(1055) |
| classic | classic(4101)、basic(1107)、traditional(421) |
| elegant | elegant(2174)、chic(1000)、luxury(405) |
| minimalist | clean(2080)、simple(1628)、minimalist(221) |
| vintage | vintage(1622)、retro(775)、old school(18) |
| formal | office(1577)、business(1063)、formal(1026) |

### use_case（21 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| gift | gift(6078)、christmas(3176)、holiday(1692) |
| party | party(4886)、evening(988)、cocktail(946) |
| everyday | daily(4347)、everyday(1879)、casual wear(726) |
| work | work(3263)、office(1577)、business(1063) |
| outdoor | outdoor(2904)、camping(571)、fishing(503) |
| swim | beach(2842)、swimming(778)、swimsuit(741) |
| running | running(2703)、jogging(623)、joggers(220) |
| winter | winter(2671)、warm(2502)、snow(576) |

### category_product_type（38 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| shirt | shirt(5909)、button down(682)、dress shirt(234) |
| blouse | top(5841)、blouse(1303)、tunic(1224) |
| dress | dress(5390)、maxi dress(419)、gown(288) |
| t_shirt | t-shirt(3320)、tee(1820)、tees(729) |
| pants | pants(3014)、trousers(345)、slacks(107) |
| jewelry | jewelry(2882)、necklace(1656)、ring(1629) |
| jeans | jeans(2509)、denim(929)、skinny jeans(318) |
| shorts | shorts(2311)、running shorts(78)、cargo shorts(69) |

### price（2 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| budget_min | premium(1887)、high end(54)、over $(8) |
| budget_max | affordable(284)、cheap(117)、under $(11) |

### attribute_aliases（6 个标准词）
| 标准词 | 同义词（带目录商品数） |
|---|---|
| size | size(13655)、fit(9888)、sizing(800) |
| style | design(8263)、style(6407)、look(4240) |
| material | material(7997)、fabric(7271)、made of(4259) |
| color | color(5191)、tone(849)、shade(150) |
| use_case | occasion(3738)、use(2829)、event(376) |
| budget | price(427)、cost(73)、budget(12) |

### 常见成分写法（百分比组合）
| 写法 | 商品数 |
|---|---|
| 100% cotton | 1106 |
| 95% polyester | 784 |
| 100% polyester | 630 |
| 50% cotton | 627 |
| 90% cotton | 617 |
| 60% cotton | 465 |
| 95% cotton | 458 |
| 95% rayon | 366 |
| 90% polyester | 261 |
| 65% polyester | 240 |
| 65% cotton | 221 |
| 92% polyester | 206 |
| 80% cotton | 191 |
| 100% leather | 180 |
| 60% polyester | 161 |

## 三、提问价值分析

### 总体（200 会话）
- 不提问基线候选池均值：**4930.7** 件

| 问法 ask_attribute | 平均披露约束数 | 缩小后候选池(均值) | 中位池 | 命中保持率 | 相对基线缩小 |
|---|---|---|---|---|---|
| other | 1.875 | 307.7 | 6 | 99.0% | 4622.9 |
| feature | 1.63 | 461.2 | 23 | 99.0% | 4469.5 |
| material | 1.025 | 2172.2 | 48 | 99.5% | 2758.5 |
| color | 0.285 | 3691.1 | 479 | 100.0% | 1239.6 |
| style | 0.085 | 4243.9 | 631 | 99.5% | 686.8 |
| size | 0.05 | 4764.3 | 697 | 100.0% | 166.4 |
| use_case | 0.015 | 4908.8 | 697 | 100.0% | 21.9 |
| category | 0.0 | 4930.7 | 748 | 100.0% | 0.0 |
| brand | 0.0 | 4930.7 | 748 | 100.0% | 0.0 |
| budget | 0.0 | 4930.7 | 748 | 100.0% | 0.0 |

### 先问什么建议
优先问 other（平均每轮把候选从 4930.7 缩小到 307.7，命中保持 0.99）；其次 feature / material。注：'other' 一次最多披露 2 条任意约束，信息量最大，通常最划算。

### 分场景
- **buying**（80 会话）：先问 `other`，候选池均值 1224.0 → 48.8；次选 `feature` / `material`
- **browsing**（80 会话）：先问 `other`，候选池均值 9316.5 → 156.2；次选 `feature` / `material`
- **intent_override**（30 会话）：先问 `other`，候选池均值 3321.9 → 64.1；次选 `feature` / `color`
- **boundary**（10 会话）：先问 `category`，候选池均值 4323.5 → 4323.5；次选 `material` / `color`