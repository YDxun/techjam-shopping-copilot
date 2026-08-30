# TechJam2026 Data Inventory + Product Dictionary + Question-Value Analysis Report

- generated at: 2026-08-28 02:18:49
- data sources: `data/catalog.jsonl` (50000 products) + `data/public_set.jsonl` (200 sessions)

## 1. Data overview

### Field coverage
| Field | Coverage |
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

### Missing & dirty data
| Issue | Count | Share |
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

### Price
- products with a price: **20.8%**; median $22.88, P25 $14.99, P75 $39.99, range $0.0-$4119.0
  -> **budget constraints must be lenient**: 79% of products have no price, so budget can never be a hard filter

### Category distribution (top second-level categories)
| Second-level category | Count |
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

### Rating distribution
| average_rating bucket | Count |
|---|---|
| 4.0-4.5 | 17104 |
| 4.5-5.0 | 9542 |
| 5.0 | 7034 |
| 3.5-4.0 | 8881 |
| 3.0-3.5 | 4542 |
| <3.0 | 2897 |

| rating_number bucket | Count |
|---|---|
| 0 | 0 |
| 1-10 | 23455 |
| 11-100 | 17224 |
| 101-1000 | 7789 |
| 1000+ | 1532 |

## 2. Product dictionary (vocab.json)

### material (72 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| polyester | polyester(10339)、100% polyester(2998)、poly(440) |
| cotton | cotton(9414)、100% cotton(3698)、cotton blend(668) |
| fabric | fabric(7271)、soft fabric(573) |
| leather | leather(6882)、faux leather(478)、genuine leather(473) |
| rubber | rubber(6616)、rubber sole(5886)、synthetic sole(1405) |
| spandex | spandex(5412)、stretch(2825)、elastane(945) |
| denim | jeans(2509)、denim(929)、jean(432) |
| mesh | mesh(2466)、mesh fabric(113)、netting(14) |

### color (45 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| black | black(6802)、black white(90)、jet black(9) |
| white | white(3319)、off-white(14)、pure white(10) |
| silver | silver(2938)、gunmetal(40)、metallic silver(22) |
| blue | blue(2670)、light blue(161)、dark blue(100) |
| gold | gold(2444)、golden(171)、metallic gold(10) |
| pattern | printed(2192)、floral(1657)、pattern(1340) |
| red | red(1916)、crimson(26)、scarlet(23) |
| grey | grey(1741)、heather grey(599)、gray(571) |

### size (34 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| s | s(28194)、small(2942) |
| shoe_size | 5(9364)、8(4290)、10(4246) |
| tall | long(7879)、tall(623) |
| numeric | 2(7408)、4(5547)、8(4290) |
| petite | short(4223)、petite(251) |
| m | m(3733)、medium(1967)、regular(1053) |
| waist_inseam | waist(3669)、inseam(547)、waist 32(28) |
| one_size | adjustable(3542)、one size(1716)、free size(65) |

### style (44 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| romantic | soft(9250)、feminine(451)、romantic(127) |
| casual | casual(7486)、everyday(1879)、weekend(362) |
| trendy | fashion(5006)、stylish(2769)、trendy(1055) |
| classic | classic(4101)、basic(1107)、traditional(421) |
| elegant | elegant(2174)、chic(1000)、luxury(405) |
| minimalist | clean(2080)、simple(1628)、minimalist(221) |
| vintage | vintage(1622)、retro(775)、old school(18) |
| formal | office(1577)、business(1063)、formal(1026) |

### use_case (21 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| gift | gift(6078)、christmas(3176)、holiday(1692) |
| party | party(4886)、evening(988)、cocktail(946) |
| everyday | daily(4347)、everyday(1879)、casual wear(726) |
| work | work(3263)、office(1577)、business(1063) |
| outdoor | outdoor(2904)、camping(571)、fishing(503) |
| swim | beach(2842)、swimming(778)、swimsuit(741) |
| running | running(2703)、jogging(623)、joggers(220) |
| winter | winter(2671)、warm(2502)、snow(576) |

### category_product_type (38 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| shirt | shirt(5909)、button down(682)、dress shirt(234) |
| blouse | top(5841)、blouse(1303)、tunic(1224) |
| dress | dress(5390)、maxi dress(419)、gown(288) |
| t_shirt | t-shirt(3320)、tee(1820)、tees(729) |
| pants | pants(3014)、trousers(345)、slacks(107) |
| jewelry | jewelry(2882)、necklace(1656)、ring(1629) |
| jeans | jeans(2509)、denim(929)、skinny jeans(318) |
| shorts | shorts(2311)、running shorts(78)、cargo shorts(69) |

### price (2 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| budget_min | premium(1887)、high end(54)、over $(8) |
| budget_max | affordable(284)、cheap(117)、under $(11) |

### attribute_aliases (6 canonical terms)
| Canonical | Synonyms (catalog product counts) |
|---|---|
| size | size(13655)、fit(9888)、sizing(800) |
| style | design(8263)、style(6407)、look(4240) |
| material | material(7997)、fabric(7271)、made of(4259) |
| color | color(5191)、tone(849)、shade(150) |
| use_case | occasion(3738)、use(2829)、event(376) |
| budget | price(427)、cost(73)、budget(12) |

### Common ingredient phrasings (percentage blends)
| Phrasing | Product count |
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

## 3. Question-value analysis

### Overall (200 sessions)
- baseline candidate-pool mean without asking: **4930.7** items

| ask_attribute | Avg disclosed constraints | Shrunk pool (mean) | Median pool | Hit retention | Shrink vs baseline |
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

### What to ask first
Ask other first (shrinks the pool from 4930.7 to 307.7 per turn on average while retaining a 0.99 hit rate); then feature / material. Note: 'other' discloses up to 2 arbitrary constraints at once, carrying the most information and usually the best value.

### Per scenario
- **buying** (80 sessions): ask `other` first, mean pool 1224.0 -> 48.8; next `feature` / `material`
- **browsing** (80 sessions): ask `other` first, mean pool 9316.5 -> 156.2; next `feature` / `material`
- **intent_override** (30 sessions): ask `other` first, mean pool 3321.9 -> 64.1; next `feature` / `color`
- **boundary** (10 sessions): ask `category` first, mean pool 4323.5 -> 4323.5; next `material` / `color`