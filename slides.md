---
# try also 'default' to start simple
theme: default
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
# background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: meeting
info: |
  ## meeting
# apply UnoCSS classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable Comark Syntax: https://comark.dev/syntax/markdown
comark: true
# duration of the presentation
duration: 35min
---

# Individual meeting 



## 2026/04/09 張嘉銘
<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
---

# What is Slidev?


Hover on the bottom-left corner to see the navigation's controls panel, [learn more](https://sli.dev/guide/ui#navigation-bar)


---
layout: default
---

# Dataset 擴充：針對 False Positive 問題

  舊版所有 sample 都含 CCTag，模型從未見過「無 CCTag 的場景」，導致 FP 過多。新版新增5,000張 negative-heavy sample，特別涵蓋複雜背景與過曝環境；並加入真實場景資料。

<div class="flex justify-center">

| Subset | Old | New | 說明 |
| :--- | ---: | ---: | :--- |
| `base_set` | 3,000 | 4,000 | 無遮擋清晰 CCTag |
| `hard_set` | 2,000 | 3,000 | 高遮擋 |
| `extreme_set` | 1,000 | 1,500 | 極端遮擋 |
| `small_set` | 1,000 | 1,500 | 小尺寸 CCTag |
| <span class="text-red-600 font-bold dark:text-red-400">**hard_negative_set**</span> | — | <span v-mark.rect.red="1">**3,000**</span> | 複雜背景、<span class="text-amber-600 dark:text-amber-400">無 CCTag</span> |
| <span class="text-red-600 font-bold dark:text-red-400">**overexposure_set**</span> | — | <span v-mark.rect.red="1">**2,000**</span> | <span class="text-orange-500">過曝場景</span> |
| <span class="text-blue-600 font-bold dark:text-blue-400">**real_world_data**</span> | — | <span v-mark.rect.blue="1">**2,086**</span> | <span class="text-blue-500">真實拍攝（5 sessions，含 279 正樣本 + 1,807 負樣本）</span> |
| **Total** | <span class="text-gray-400">7,000</span> | <span class="text-green-600 font-bold dark:text-green-400 text-lg">17,086</span> | |

</div>






---
layout: center
class: text-center
---

# END
