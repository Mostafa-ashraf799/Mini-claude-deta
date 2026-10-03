# mini_claude.py — نموذج لغوي من الصفر (Dense / MoE / Towers + أدوات + صور اختيارية)

ملف واحد (`mini_claude.py`) فيه كل حاجة: التوكنايزر، المعمارية، التدريب (pretrain + SFT)، الـ chat، الأدوات، ودعم الصور.
المتطلبات: `pip install torch tokenizers numpy pillow` (Pillow مطلوب بس لو هتستخدم الصور).

> **أمانة في الكلام ده:** كل حاجة اتجربت على بيانات تجريبية صغيرة جداً وعلى CPU عشان نتأكد إن الكود شغال صح (الـ `selftest` + تجارب end-to-end). مفيش حد دلوقتي ضامن جودة نموذج كبير — ده بيتحدد بالبيانات والـ GPU والوقت. النقاط اللي مش مضمونة مكتوبة صراحة في قسم "حدود وملاحظات".

---

## 1) الفكرة العامة

```
نص خام (كتب/شرح)  ──prep──►  tokenizer + train.bin  ──train──►  ckpt.pt   (النموذج بيتعلم اللغة والمعلومات)
أسئلة وأجوبة JSONL ─prep_sft─► sft_*.bin             ──sft────►  sft_ckpt.pt (النموذج بيتعلم يجاوب/يحل خطوة خطوة/يستخدم أدوات/يشوف صور)
                                                      chat / generate ◄── بتكلم النموذج
```

كل مرحلة بتكتب في فولدر `--out` واحد (بتختاره انت)، فالفولدر ده هو "مشروعك".

---

## 2) المعماريات التلاتة (انت بتختار بـ `--arch`)

كل واحدة شغالة **لوحدها** تماماً، ومفيش حاجة من التانية بتتحمّل أو بتتحسب.

| `--arch` | إيه هي | الباراميترز الكلية | الشغال لكل توكن | الذاكرة | ميزتها |
|---|---|---|---|---|---|
| `dense` | كل التوكنز بتعدّي على كل الطبقات (Transformer عادي) | N | N | الأقل | الأبسط والأضمن في التدريب |
| `moe` | في كل طبقة (أو كل `moe_every`) عدد من الخبراء، والراوتر بيختار `topk` خبراء **لكل توكن** | ≈ 2× dense (بإعدادات افتراضية) | ≈ N | أعلى | سعة معرفة أكبر بنفس تكلفة الحساب |
| `towers` | **الأبراج**: كل خبير = سلسلة طبقات كاملة (برج). الراوتر بيختار البرج المناسب للسؤال، وبس البرج ده بيشتغل | T × N | N (برج واحد) | أعلى (كل الأبراج في الذاكرة) | أسرع في التشغيل + تخصص حقيقي لكل مجال |

### Dense
```bash
python3 mini_claude.py train --out proj --arch dense --size base
```

### MoE (لوحده)
```bash
python3 mini_claude.py train --out proj --arch moe --size base \
    --moe_experts 16 --moe_topk 2 --moe_shared 1 --moe_div 4 --moe_every 2
```
- `--moe_experts`: أي عدد خبراء (8، 16، 64…).
- `--moe_topk`: كام خبير يشتغلوا لكل توكن.
- `--moe_shared`: خبراء ثابتين بيشتغلوا دايماً (بيجمعوا المعرفة العامة).
- `--moe_div`: حجم الخبير = FFN ÷ الرقم (4 أو 8 = خبراء أصغر وأدق وأكتر).
- `--moe_every`: MoE كل كام طبقة (2 = طبقة آه وطبقة لأ).
- فيه aux loss لتوزيع الحمل على الخبراء (`aux_coef`) تلقائي.

### Towers (الطريقة الجديدة اللي طلبتها)
فكرتها: بدل ما كل توكن يروح لخبير مختلف في كل طبقة (MoE)، **السؤال كله** بيروح لبرج واحد (سلسلة طبقات كاملة: فيزياء، كيمياء، رياضيات…). الباقي مش بيشتغل أصلاً → أسرع.

- Embedding والـ head مشتركين بين الأبراج، وكل برج له `norm` نهائي خاص بيه.
- الراوتر = MLP صغير على متوسط embedding للسؤال، بيطلّع احتمال لكل برج.
- **الراوتر بيختار لكل تسلسل (sequence) مش لكل توكن** — ده قرار مقصود: لأن لو كل توكن راح لبرج مختلف، الـ attention والـ KV cache مش هيبقوا متوافقين بين الأبراج.
- التدريب بـ **لابلز المجال**: كل مثال لازم يتعلّم عليه البرج بتاعه (hard routing) + loss إضافي للراوتر (`--router_coef`).
- وقت التشغيل: `--tower_topk 1` (الأسرع: برج واحد)، أو 2+ فيتم دمج log-probs أكتر من برج (أبطأ، أحياناً أفضل في المسائل المشتركة).
- ممكن تجبر برج معين: `--tower physics`.

تجهيز بيانات towers: **فولدر لكل مجال**
```
data/
  physics/    *.txt
  chemistry/  *.txt
  math/       *.txt
```
```bash
python3 mini_claude.py prep  --data data --out proj --vocab 8000 --by_domain
python3 mini_claude.py train --out proj --arch towers --size base          # عدد الأبراج = عدد المجالات تلقائي
# أو: --towers 5 لو عايز عدد مختلف
```
المجالات بتتحفظ في `proj/domains.json`. أمثلة الـ SFT لازم فيها حقل `"domain"` (أو استخدم `--default_domain`).

#### ازاي أعمل مشروع "المايسترو + متخصصين" بالأبراج؟
ده أقرب شكل لفكرتك: كل برج متخصص، والراوتر هو "المايسترو". السؤال الواضح يروح لبرج واحد، والمسائل المركبة تستخدم `--tower_topk 2` أو 3.
ملحوظة: ده نموذج **واحد** فيه الأبراج، مش نماذج منفصلة بتتكلم مع بعض.

---

## 3) دعم الصور (اختياري — انت بتقرر)

- **مغلق افتراضياً:** `--vision off` → مفيش أي كود صور بيتحمّل ولا باراميترز زيادة.
- **تفعيل:** `--vision on` وقت `train` (لازم من البداية، لأن المشفّر جزء من الأوزان).
- **ينفع مع أي معمارية** (dense / moe / towers).

```bash
python3 mini_claude.py train --out proj --arch dense --size base --vision on
```
الأحجام الافتراضية للصور (قابلة للتعديل بـ `--img_size --patch --v_dim --v_layers --v_heads --pool`):

| الحجم | صورة | patch | v_dim | طبقات | توكنز الصورة |
|---|---|---|---|---|---|
| tiny / small | 64 | 16 | 192 | 3 | 16 |
| base / large | 224 | 16 | 384 | 6 | 49 |
| xl … 5b | 224 | 14 | 768 | 12 | 64 |
| 7b+ | 224 | 14 | 1024 | 16 | 64 |

إزاي بيشتغل: الصورة → تتقسم patches → ViT encoder → projector (MLP) → `n_img_tokens` متجه بيحل محل توكن `<|image|>` في الـ embeddings. الصور بتتدرّب **في مرحلة SFT بس** (مثال = صورة واحدة + سؤال + إجابة). الـ pretrain على النص بيسيب المشفّر من غير استخدام.

مثال SFT بصورة:
```json
{"image": "images/circuit1.png", "prompt": "ما قيمة المقاومة المكافئة؟", "response": "الخطوة 1: ... الإجابة: 6 أوم"}
```
```bash
python3 mini_claude.py prep_sft --out proj --sft_data sft.jsonl --image_root /path/to/images_folder
python3 mini_claude.py sft --out proj --vision_mix 0.5
python3 mini_claude.py chat --out proj --image photo.png        # أو جوه الـ chat: /image PATH
```
**نتيجة التجربة الحقيقية (مهم تقراها):** على صور اصطناعية بسيطة (أشكال ملونة) بنموذج tiny:
- **اللون:** 100% على 60 صورة جديدة (ومع صورة فاضية نزلت للصدفة) → يعني الصورة فعلاً بتؤثر والـ pipeline سليم.
- **الشكل (دائرة/مربع/…):** حوالي 48% = الصدفة، حتى بعد تجربة patch أدق (4×4). السبب إن النموذج صغير جداً وبيانات التجربة قليلة. **ده مش عيب في الكود** (اتأكدت إن الـ patchify سليم والـ gradients واصلة)، لكن ده معناه إن فهم صور حقيقي (رسومات فيزياء، دوائر كهربية…) محتاج: نموذج أكبر، آلاف/ملايين أمثلة صور، وغالباً **مشفّر صور متدرّب مسبقاً** (مش مدعوم في الكود ده حالياً — المشفّر بيتدرّب من الصفر).

---

## 4) الأدوات (Tool Use)

- بتشتغل تلقائياً في الأحجام **2b وفوق** (`--tools auto`)، وممكن تفرضها `--tools on` / تقفلها `--tools off`.
- النموذج بيكتب: `<|tool_call|>{"name": "...", "arguments": {...}}<|/tool_call|>`، الـ chat بينفّذ الأداة ويرجّع النتيجة برول `tool`، وبعدين النموذج يكمّل (حد أقصى `--max_tool_rounds`).
- مدمجة: `calculator` (آمنة بـ AST، مفيش eval) و `get_time`.
- أدواتك الخاصة: ملف بايثون فيه
```python
# my_tools.py
TOOLS = {
  "square": {"fn": lambda x: x*x, "description": "يربّع رقم",
             "parameters": {"x": "number"}}
}
```
```bash
python3 mini_claude.py chat --out proj --tools_file my_tools.py
```
- أمثلة الأدوات في SFT بتتجاهل تلقائياً لو النموذج مش بيدعم أدوات (وبيتطبع تحذير بعددها). `--force_tools` بتضمها غصب.
- النموذج الصغير هيحتاج **أمثلة أدوات كتير** عشان يتعلم يستخدمها صح. الكود بيوفّر الآلية، مش الذكاء.

---

## 5) ملف التدريب — إيه نوعه وشكله؟

### أ) Pretrain (المرحلة الأولى): ملفات **نصوص عادية `.txt`** (UTF-8)
- ده "شرح المعلومات": كتب، دروس، شرح قوانين، قصص، تعريفات. مفيش تنسيق معين، مجرد نص.
- مثال لمنهج مدرسي: شرح درس الحركة، قوانين نيوتن، الجدول الدوري… كله كنصوص شرح.
- `prep` بيدرّب tokenizer (BPE) من النصوص دي ويحوّلها لـ `train.bin` (+ `val` تلقائي).
- لـ towers: فولدر لكل مجال + `--by_domain`.
- ينفع تحط ملفات `.txt` كتير في فولدر، الكود بيقراها كلها.

```bash
python3 mini_claude.py prep --data my_texts/ --out proj --vocab 16000
```
> لو غيّرت التوكنايزر أو فولدر `--out` لازم تعيد `prep` (التوكنايزر والـ bin لازم يكونوا من نفس العملية).

### ب) SFT (المرحلة التانية): ملف **JSONL** — كل سطر مثال JSON
الكود بيفهم 3 صيغ لنفس الغرض:

**1. prompt / response** (الأبسط — وده المناسب للمنهج):
```json
{"prompt": "حل: 2x + 6 = 14", "response": "الخطوة 1: نطرح 6 من الطرفين: 2x = 8\nالخطوة 2: نقسم على 2: x = 4\nالإجابة: x = 4", "domain": "math"}
```
**2. instruction / input / output** (صيغة Alpaca):
```json
{"instruction": "حل المعادلة", "input": "2x+6=14", "output": "..."}
```
**3. messages** (محادثة كاملة بأدوار system/user/assistant/tool):
```json
{"messages": [
  {"role": "system", "content": "أنت مدرّس رياضيات."},
  {"role": "user", "content": "كام 17*23؟"},
  {"role": "assistant", "tool_calls": [{"name": "calculator", "arguments": {"expression": "17*23"}}]},
  {"role": "tool", "content": "391"},
  {"role": "assistant", "content": "17 × 23 = 391"}
]}
```
حقول اختيارية: `domain` (للـ towers)، `system`، `tools`، `image`.

**عشان الإجابة تبقى "خطوات الحل":** اكتب الخطوات جوه `response` بالنص (زي المثال). النموذج بيتعلم يقلّد الشكل ده. مفيش قالب سحري تاني.

**الـ loss** في SFT بيتحسب **على رد المساعد بس** (السؤال ورسالة النظام مش بيتحاسبوا) — فالنموذج بيتعلم يجاوب، مش يحفظ الأسئلة.

### ج) ترتيب الشغل الكامل
```bash
# 1) نصوص → tokenizer + bins
python3 mini_claude.py prep --data texts/ --out proj --vocab 16000 [--by_domain]
# 2) pretrain
python3 mini_claude.py train --out proj --size base --arch dense|moe|towers [--vision on] --steps 20000
# 3) جهّز SFT
python3 mini_claude.py prep_sft --out proj --sft_data sft.jsonl [--image_root imgs/] [--default_domain math]
# 4) SFT
python3 mini_claude.py sft --out proj --steps 2000
# 5) كلّمه
python3 mini_claude.py chat --out proj [--image x.png] [--tower physics] [--tower_topk 2]
```
`prep_sft` بيقرا `proj/ckpt.pt` عشان يعرف النموذج بيدعم أدوات/صور/كام ctx، وبيطبع تحذيرات (أمثلة اتخطّت أو طويلة).

---

## 6) الأحجام والذاكرة

`python3 mini_claude.py info` بيطبع الجدول ده بالأرقام المحسوبة (متأكد منها مقابل العدّ الفعلي للباراميترز):

| الحجم | dense | MoE (كلي) | Towers (كلي، 4 أبراج) | تدريب dense* | تشغيل fp16 | أدوات |
|---|---|---|---|---|---|---|
| tiny | 0.02B | 0.03B | 0.05B | – | – | لا |
| small | 0.04B | 0.07B | 0.11B | 1GB | 0.1GB | لا |
| base | 0.10B | 0.19B | 0.33B | 2GB | 0.2GB | لا |
| large | 0.30B | 0.60B | 1.10B | 5GB | 0.6GB | لا |
| xl | 0.95B | 1.97B | 3.61B | 15GB | 1.9GB | لا |
| 1.5b | 1.48B | 3.11B | 5.74B | 24GB | 3GB | لا |
| 2b | 2.01B | 4.22B | 7.81B | 32GB | 4GB | نعم |
| 3b | 3.02B | 6.19B | 11.79B | 48GB | 6GB | نعم |
| 4b | 4.02B | 8.33B | 15.81B | 64GB | 8GB | نعم |
| 5b | 4.99B | 10.56B | 19.62B | 80GB | 10GB | نعم |
| 7b | 7.01B | 14.68B | 27.67B | 112GB | 14GB | نعم |
| 8b | 8.07B | 16.95B | 31.91B | 129GB | 16GB | نعم |
| 9b | 8.98B | 18.66B | 35.50B | 144GB | 18GB | نعم |

\* التدريب ≈ **16 بايت لكل باراميتر** (أوزان + gradients + Adam) **على كل كارت** — الكود **مفيهوش sharding (FSDP/ZeRO)**. يعني عملياً على كارت واحد (Kaggle T4 16GB / P100 16GB) تقدر تدرّب لحد حوالي **small/base/large** (ولحد xl بصعوبة مع `--grad_ckpt` وbatch صغير). الأحجام 1.5b وفوق موجودة كـ presets لكن **محتاجة كروت أكبر أو إضافة sharding** — الكود بيعمل **memory guard** ويرفض يبدأ لو الذاكرة مش كفاية (إلا لو `--force`).

الـ towers/MoE بيكبّروا الأوزان الكلية (الذاكرة) مش الحساب لكل توكن.

---

## 7) على Kaggle

- GPU واحد: شغّل الأمر عادي. جلسة Kaggle ليها حد وقت → استخدم `--max_hours 8` و`--save_interval`؛ لو الجلسة وقفت أعد نفس الأمر وهو **بيكمّل من آخر checkpoint** (resume تلقائي).
- أكتر من GPU (T4×2): `python3 -m torch.distributed.run --nproc_per_node=2 mini_claude.py train ...` (DDP — بيعمل نسخة كاملة على كل كارت، مش بيقسّم الذاكرة).
- fp16/bf16 + GradScaler تلقائي حسب الكارت. `--grad_ckpt` بيوفر ذاكرة، `--compile` اختياري.
- بيانات الدرس الأول: ابدأ بـ `--size small`/`base` و10–100MB نصوص لتتأكد إن كل حاجة شغالة قبل أي حاجة كبيرة.
- ⚠️ الـ DDP لـ towers والصور **اتكتب فيه الكود (`find_unused_parameters=True`) لكن مجربتوش على أكتر من كارت هنا** — جرّبه بـ 2 كارت وبـ `--steps` صغير الأول.

---

## 8) أجزاء الكود (خريطة الملف)

| الجزء | الوصف |
|---|---|
| `PRESETS`, `Config`, `vision_defaults` | الأحجام والإعدادات (أي checkpoint بيحفظ الـ Config جواه) |
| `estimate_params`, `count_params` | حساب الباراميترز الكلية/النشطة (لـ `info` وللـ memory guard) |
| `RMSNorm`, `Attention` | RMSNorm + RoPE + GQA (`n_kv_head < n_head`) + KV cache |
| `SwiGLU`, `MoE`, `Block` | شبكة التغذية، الـ MoE (تجميع التوكنز لكل خبير بالترتيب، shared experts، aux loss)، وكتلة الطبقة |
| `ViTBlock`, `VisionEncoder` | مشفّر الصور + الـ projector |
| `GPT` | النموذج الكامل: dense/moe بـ `self.blocks`، towers بـ `self.towers` + `self.norms` + `self.router`؛ `forward(idx, targets, caches, pos0, images, tower)` |
| الأدوات | `tool_calculator`, `BUILTIN_TOOLS`, `load_tools`, `parse_calls`, `run_tool` |
| القوالب والبيانات | `render_message`, `build_prompt`, `encode_chat`, `normalize_example`, `prep`, `prep_sft`, `Shard`, `VisionData` |
| التدريب | `setup_dist` (DDP)، `make_cfg`، `evaluate`، `run_train` (cosine LR + warmup، AdamW، resume، `max_hours`) |
| التوليد والـ chat | `sample` (top-k/top-p/rep_pen، streaming)، `resolve_tower`، `chat` (أدوات، صور، `/image`, `/reset`, `/exit`) |
| الأوامر | `info`, `selftest`, `prep`, `train`, `prep_sft`, `sft`, `generate`, `chat` |

قالب المحادثة: `<|role|>\n المحتوى <|end|>` والـ loss على جسم رد المساعد فقط. توكنز خاصة: `<|eos|> <|system|> <|user|> <|assistant|> <|end|> <|tool|> <|tool_call|> <|/tool_call|> <|image|>`.

---

## 9) مرجع الـ arguments (مختصر)

**train / sft:** `--out` `--size` `--ctx` `--batch` `--accum` `--steps` `--lr` `--warmup` `--eval_interval` `--eval_iters` `--save_interval` `--log_interval` `--max_hours` `--grad_ckpt` `--compile` `--force` `--domain_balance {auto,uniform,size}` `--vision_mix` (sft)
**train فقط:** `--arch {auto,dense,moe,towers}` `--tools {auto,on,off}` `--vision {on,off}` `--img_size --patch --v_dim --v_layers --v_heads --pool` `--moe_experts --moe_topk --moe_every --moe_shared --moe_div` `--towers --router_coef`
**generate / chat:** `--out --ckpt --max_new --temp --top_k --top_p --rep_pen --tower --tower_topk` + (chat) `--system --once --image --no_tools --tools_file --max_tool_rounds`
**prep:** `--data --out --vocab --by_domain` — **prep_sft:** `--out --sft_data --force_tools --image_root --default_domain`

---

## 10) اللي اتجرّب فعلاً

- `selftest`: كل المعماريات (dense، moe 4 خبير، moe 16 + shared + div4، towers 3، صور بـ pool=1 و pool=2): KV cache مطابق للحساب الكامل، حفظ batch صغير (loss ينزل من ~4.9 إلى ~0.03)، الراوتر يصنّف صح، الصور بتأثر على الناتج والـ gradients بتوصل للمشفّر، وعدّ الباراميترز الفعلي = التقدير لـ 13 حجم.
- end-to-end على بيانات صغيرة: pretrain → SFT → chat للـ dense وMoE وtowers (الراوتر 100% على مجالات اصطناعية، والبرج الصح هو اللي بيجاوب، وإجبار برج غلط بيدي إجابة المجال الغلط)، والأدوات (الحاسبة).
- ❌ ما اتجرّبش: تدريب حقيقي على GPU، أحجام أكبر من tiny/small بالتدريب الفعلي، DDP متعدد الكروت، جودة النموذج على بيانات منهج حقيقية.

## 11) حدود وملاحظات

1. **Towers = routing لكل سؤال**، مش لكل توكن، ومحتاجة بيانات موسومة بالمجال. مش هتشتغل كويس لو المجالات متداخلة جداً من غير تدريب كفاية للراوتر.
2. الأبراج لا تقلل الذاكرة (كلها محمّلة) — بتقلل **وقت الحساب** بس.
3. طول السياق للـ tiny..large صغير؛ لو الـ system prompt بتاع الأدوات أطول من الـ ctx بيتقص → الكود بيحذّرك. الأحجام ≥2b عندها ctx 4096.
4. الصور: بتتعلم اللون، الشكل محتاج حجم/بيانات أكبر (انظر القسم 3).
5. مفيش sharding → الأحجام الكبيرة (≥1.5b) مش هتتدرّب على كارت واحد.
6. جودة الإجابات في العلوم بتعتمد على جودة وكمية بيانات المنهج وخطوات الحل — النموذج الصغير ممكن "يقلّد" شكل الحل ويغلط في الحساب (وده سبب وجود أداة الحاسبة).
7. لو غيّرت `--vocab` أو الفولدر: أعد `prep` و`prep_sft`.
8. للتجارب السريعة: `--size tiny` أو `small` مع `--steps` صغير.
