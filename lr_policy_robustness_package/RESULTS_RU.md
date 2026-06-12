# Пакет воспроизводимых результатов LR Policy Robustness

Это самодостаточный артефакт по экспериментам CIFAR-100 Learning-from-Teaching с обучаемой политикой learning rate. Внутри лежат снимок кода, oracle-данные для обучения политики, кеш CIFAR-100, policy checkpoints, финальные checkpoints моделей, сырые логи, распарсенные таблицы, графики и ноутбуки для проверки и воспроизведения результатов.

## Прямые ссылки на ноутбуки

- [notebooks/inference_and_figures.ipynb](notebooks/inference_and_figures.ipynb) - headline-результат PatchTST seed 3, кривые обучения, траектория LR.
- [notebooks/robust12h_summary.ipynb](notebooks/robust12h_summary.ipynb) - сводка robustness-кампании, paired deltas и winning seeds.

## Executive Summary

Обучаемая LR-policy не является универсальной заменой cosine scheduler. Самый корректный вывод уже:

- Лучший одиночный запуск: `PatchTST output ensemble top5 blend075`, seed 3, 60 эпох: **70.70** против cosine seed 3 **70.45**.
- На нескольких seed PatchTST в основном **сравним** с cosine, а не уверенно лучше: top5 blend075 дает **70.08 +/- 0.59** против cosine **70.20 +/- 0.23** на 60 эпохах.
- Самый чистый robustness-win - мягкий LR-down shock на epoch 20: `shock_lr=0.10`, PatchTST **70.17 +/- 0.38** против cosine **69.45 +/- 0.52**, delta **+0.72 pp**.
- На label noise, input noise, compound perturbations и экстремальных LR shocks cosine остается сильнее или статистически сравним.
- Ценность подхода лучше описывать как **trajectory repair** в отдельных schedule-perturbation сценариях, а не как новый универсальный scheduler.

Промежуточный вывод: `residual_log` параметризация оказалась правильным safety prior. Она держит поведение близко к cosine, предотвращает катастрофический провал raw-LR политики и дает несколько полезных robustness-кейсов.

## Карта исследования и всех результатов

Все исходные markdown-отчеты из исследования сохранены в `reports/source_results/`. Таблица ниже показывает, где внутри пакета лежит каждый этап и какой вывод из него следует цитировать.

| Этап | Source report в пакете | Основной результат | Промежуточный вывод |
|---|---|---|---|
| Local smoke test / raw LR | `reports/source_results/hour_gpu_summary.md`, `reports/source_results/final_resume.md` | Raw dynamic LR: **60.23** final vs cosine **69.13**. Safe cosine multiplier: **69.17 +/- 0.14** на 6 seed. | Прямое предсказание LR ломается, потому что late LR остается слишком высоким; нужны ограниченные schedule-relative outputs. |
| Overnight policy baselines | `reports/source_results/overnight_summary.md` | Current GRU policy: **34.45-44.34**; modular policy: **68.29-69.05**; cosine seeds около **69.35-70.11**. | Одна смена архитектуры не решает проблему target/output-параметризации. |
| Six-hour multiplier follow-up | `reports/source_results/six_hour_policy_summary.md` | Лучшие варианты около **69.33-69.44**, ниже cosine reference **69.81 +/- 0.33**. | Multiplier tuning возвращает baseline-level behavior, но не дает clear win. |
| Curve / MPC policy POC | `reports/source_results/2h_curve/RESULTS.md` | Curve frozen **32.73**, curve online **31.73**, cosine ref **70.64** на seed 400. | Предсказание curve может улучшать MSE, но argmin уходит в разрушительно высокий LR. |
| Curve optimized follow-up | `reports/source_results/1h_curve_opt/RESULTS.md` | Лучший optimized curve variant peak **55.03**, final/last observed **50.16**, все еще далеко ниже cosine **69.81 +/- 0.33**. | Capping/blending улучшает POC примерно на +23 pp, но curve-направление остается null. |
| Literature-driven proposals | `reports/source_results/trainable_lr_literature_review.md` | Ranked proposals P1-P7: residual-log head, AdaLRS veto, schedule-free/hypergrad baselines, bandit reformulation, sharpness features. | Рабочее направление - bounded residual control over cosine, а не unconstrained LR prediction. |
| Night10h + catchup | `reports/source_results/night10h/RESULTS.md`, `reports/source_results/RESULTS_combined.md` | Residual-log GRU: **69.31 +/- 0.43** на 6 seed против cosine **69.81 +/- 0.33**. Hypergrad/SF baselines сильно ниже cosine. | P3 residual-log - первый tight learnable-policy режим, но среднее все еще чуть ниже cosine. |
| Full pipeline horizons | `reports/source_results/RESULTS_full_pipeline.md` | 35ep: -0.50 pp; 60ep final: -0.39 pp; 60ep last-10: **+0.05 pp**; warm restart дает 2.5x меньший transient drop. | Residual-log переносится между horizons и улучшает shock tolerance, но не среднюю final accuracy. |
| SOTA architecture sweep | `reports/source_results/advanced_industry/summary.md` | PatchTST offline MSE **0.258**, 6,984 params; deployment **69.68 +/- 0.16** на 35ep. | PatchTST - лучший компактный backbone для LR-policy. |
| PatchTST follow-ups | `reports/source_results/patchtst_followups_current/current_summary.md` | Fixed NBeats **69.64**, DLinear **69.56** на 35ep; PatchTST ext50 около **69.4-69.6** while running. | Fixed-window архитектуры можно запускать после padding fixes, но PatchTST практичнее. |
| Blend-tuning screen | `reports/source_results/blendtune3h_combined/SUMMARY.md` | Blend080 single seed reached **70.55**, +0.10 vs cosine seed3, но ниже previous **70.70**. | Только умеренные cosine blends у PatchTST иногда бьют cosine. |
| Beat-cos campaign | `reports/source_results/beatcos_combined/SUMMARY.md` | Top5 blend075: **70.083 +/- 0.592** vs cosine **70.197 +/- 0.234**; seed3 reaches **70.70**. | Headline win реален, но seed-sensitive; multi-seed mean tied/slightly below cosine. |
| Improve5h postmortem | `reports/source_results/improve5h_combined/SUMMARY.md`, `reports/source_results/improve5h_combined/analysis/report.md` | Best improve5h run **70.29**, ниже cosine seed3 **70.45** и prior PatchTST **70.70**. | Output ensembling схлопывает `policy_pred` почти в константное LR attenuation, а не adaptive control. |
| External adaptive LR comparison | `reports/source_results/sota_lr_comparison/SUMMARY.md`, `reports/source_results/sota_lr_comparison/analysis/report.md` | Best external comparator AdaLRS default: **67.41 +/- 0.82**, -2.79 pp vs cosine mean. | Generic adaptive LR controllers не бьют cosine в этом LoT setup. |
| Robust12h campaign | `reports/source_results/robust12h/REPORT.md`, `reports/robust12h_PACKAGE_SUMMARY.md` | 167 completed runs. Лучший paired robust cell: PatchTST `shock_lr=0.10`, **+0.72 pp** vs cosine. | Learned LR policy полезна как targeted trajectory repair, особенно после mild LR-down shock. |

Промежуточный вывод: история исследования похожа на funnel. Raw LR, curve argmin, schedule-free, hypergrad, bandit-style controls и широкий blend tuning в основном проваливаются; выживает более узкий claim - bounded residual-log PatchTST как conservative repair controller поверх cosine.

## Метод

Headline policy - output ensemble из пяти PatchTST residual-log LR policies:

```text
policy_lr = cosine_lr * exp(clip(mean(policy_outputs), -1, 1))
next_lr = 0.25 * policy_lr + 0.75 * cosine_lr
```

Первые 10 эпох используются как cosine warmup, затем включаются policy corrections. Ensemble checkpoints лежат в `checkpoints/policies_beatcos/`.

Промежуточный вывод: identity prior критичен. Если policy не уверена, output около нуля означает "используй cosine"; поэтому OOD-horizon results остаются близкими к baseline, а не collapse.

## Применявшиеся методы и архитектуры

| Метод / архитектура | Роль в исследовании | Краткое описание архитектуры / правила | Вывод по результатам |
|---|---|---|---|
| `cosine` | Главный hand-designed baseline | Детерминированный cosine annealing от initial LR к minimum LR на заданном horizon. Без learned state. | Самый сильный default baseline; большинство learned variants либо match, либо проигрывают. |
| `raw_lr` dynamic policy | Первая learned-LR baseline | GRU policy получает window из landscape/student features и напрямую регрессирует scalar LR. | Провал: **60.23** vs cosine **69.13** в local smoke test, потому что late LR остается слишком высоким. |
| `cosine_multiplier` | Более безопасная ранняя learned policy | Learned scalar multiplier поверх cosine, с caps/blending, чтобы держать LR близко к baseline schedule. | Вернула baseline-level behavior, около **69.17 +/- 0.14**, но без clear win. |
| Modular multiplier policy | Architecture ablation | MLP/modular policy по landscape и latent student statistics; предсказывает bounded schedule-relative LR. | Осталась около **69.3-69.4**, ниже cosine reference. |
| `residual_log` GRU | Главная safe GRU policy | GRU по feature windows; output clipped to `[-1, 1]` и декодируется как `cosine_lr * exp(output)`. | Первая стабильная learned policy: **69.31 +/- 0.43** на 35ep; tie на 60ep last-10 average. |
| `curve_argmin` / CurveLRPolicy | Multi-step landscape-forecast policy | Neural policy предсказывает полную 30-point loss curve для candidate LRs; deployment выбирает argmin или softened argmin. | Катастрофа без сильных constraints: curve MSE улучшается, но argmin уходит в разрушительно высокий LR. |
| Curve policy with cap/blend/online MPC | Stabilized curve follow-up | Та же curve forecast head, но LR capped, blended with cosine и иногда обновляется online по curve MSE. | Улучшила результат с ~32% до peak ~55%, но все еще далеко ниже cosine. |
| PatchTST residual-log | Лучший compact policy backbone | Patch-based time-series Transformer по коротким feature windows; residual-log output over cosine. | Лучший backbone: offline MSE **0.258**, 6,984 params, 35ep deployment **69.68 +/- 0.16**. |
| PatchTST output ensemble | Headline learned policy | Mean из пяти PatchTST outputs, затем clipped residual-log transform и `0.25 policy + 0.75 cosine` blend. | Лучший single run **70.70** vs cosine seed3 **70.45**; multi-seed mean tied/slightly below cosine. |
| PatchTST checkpoint soup | Ensemble/averaging ablation | Усреднение/soup нескольких PatchTST checkpoints перед residual-log deployment. | Стабильно, но не beat cosine mean; лучшие soup variants около **69.9-70.0**. |
| PatchTST zero-mean EMA | Bias-removal ablation | Вычитает running mean из policy output перед residual-log correction. | Не помогло: **69.72 +/- 0.31** в robust12h open-item check. |
| TCN policy | Time-series architecture ablation | Temporal convolutional network по feature windows, residual-log output. | Запустилась успешно и дала **69.37** на одном 35ep seed; ниже PatchTST. |
| NBeats policy | Time-series architecture ablation | Deep block-based forecast architecture, адаптированная к LR-policy features. | Сначала падала из-за fixed-window assumptions; после fixes около **69.64** на одном 35ep seed. |
| DLinear policy | Lightweight linear time-series ablation | Decomposition/linear time-series head over policy windows. | Сначала падала из-за fixed-window assumptions; после fixes около **69.56** на одном 35ep seed. |
| Schedule-Free SGD / AdamW | External adaptive optimizer baseline | Schedule-free optimizer family with iterate averaging; без learned LR policy. | Сильный negative control: далеко ниже cosine в night10h и robust12h sweeps. |
| Hypergradient / Hypergrad-HB | Online LR adaptation baseline | LR обновляется hypergradient-style сигналом из loss/probe dynamics, без offline policy training. | Не научился корректно decay LR; сильно ниже cosine. |
| AdaLRS-style veto / AdaLRS comparators | Loss-guided adaptive LR baseline | Использует loss trend / velocity-style safeguards для accept/reject/clamp LR changes. | Veto не спас curve policy; external AdaLRS был лучшим SOTA comparator, но все равно **-2.79 pp** vs cosine mean. |
| Bandit EXP3 / UCB | Discrete LR-grid baseline | LR selection как non-stationary bandit over candidate LR arms. | Очень нестабильно и сильно ниже cosine в SOTA comparison. |
| Robust perturbation protocols | Stress tests, не optimizer | LR shocks, label noise, input noise, compound noise+shock, delayed noise onset, 90ep OOD horizon. | Нашли полезную область: PatchTST чинит mild LR-down shock, но не решает noisy data или extreme shocks. |

Промежуточный вывод: поиск метода сузился от unconstrained LR prediction и curve argmin к bounded residual control over cosine. PatchTST делает controller компактнее и стабильнее, но он остается скорее targeted repair mechanism, а не universal scheduler.

## Full Pipeline Results

Основные deployment-horizon эксперименты для residual-log family:

| Horizon / setup | Cosine | Learnable policy | Delta | Главный вывод |
|---|---:|---:|---:|---|
| 35 epochs from scratch, 6 seeds | 69.81 +/- 0.33 | GRU residual-log 69.31 +/- 0.43 | -0.50 | match, но ниже cosine |
| 50 epoch warm-restart, LR jump at e36 | drop около -18 pp, recovery около 10 epochs | drop около -7 pp, recovery около 5 epochs | 2.5x smaller transient | residual policy более shock tolerant |
| 60 epochs from scratch, 3 seeds | 70.20 +/- 0.23 | GRU residual-log 69.81 +/- 0.21 | -0.39 | near-cosine final accuracy |
| 60 epochs, last-10 average | 68.42 +/- 0.35 | GRU residual-log 68.47 +/- 0.57 | +0.05 | smoothed metric tied |

Промежуточный вывод: на train horizon и OOD horizon policy не улучшает mean final accuracy, но держится близко к cosine. Положительный сигнал - robustness к schedule shocks.

## PatchTST Architecture and Beat-Cosine Campaign

PatchTST заменил исходный GRU backbone для LR-policy. Он намного меньше и лучше offline.

| Policy backbone | Parameters | Offline test MSE | Deployment result |
|---|---:|---:|---|
| PatchTST | 6,984 | 0.258 | strongest backbone |
| TCN | 115,713 | 0.288 | 69.37 on one seed |
| GRU baseline | 349,332 | 0.307 | 69.31 +/- 0.43 at 35ep |
| NBeats | 1,922,679 | 0.321 | overfit / deploy crash in early tests |
| DLinear | 253 | 0.362 | deploy crash in early tests |

60 epoch PatchTST campaign:

| Method | n | Final mean | Final std | Last-10 mean | Delta vs cosine final |
|---|---:|---:|---:|---:|---:|
| cosine | 3 | 70.197 | 0.234 | 68.425 | 0.000 |
| patchtst_output_ens_top5_blend075 | 3 | 70.083 | 0.592 | 68.590 | -0.113 |
| patchtst_output_ens_top5_blend090_ema03 | 3 | 70.003 | 0.214 | 67.977 | -0.193 |
| patchtst_soup_top5_blend090 | 3 | 69.943 | 0.146 | 68.323 | -0.253 |
| residual_log GRU reference | 3 | 69.807 | 0.205 | 68.473 | -0.390 |
| patchtst_plain | 3 | 69.627 | 0.192 | 67.499 | -0.570 |

Важный seed-level результат:

| Method | Seed | Final acc | Final loss | Last-10 acc |
|---|---:|---:|---:|---:|
| cosine | 3 | 70.45 | 1.113 | 68.83 |
| patchtst_output_ens_top5_blend075 | 3 | **70.70** | **1.093** | **69.12** |

Промежуточный вывод: PatchTST - лучший policy backbone, но 70.70 - single-seed win. Multi-seed mean tied/slightly below cosine.

## Robust12h Campaign

Robustness campaign запускалась на трех A100 серверах и просканировала **167 completed runs**:

- `stars`: LR shocks и no-shock controls.
- `industry`: label noise, input noise и compound label-noise plus LR-shock perturbations.
- `mlspace`: open-item checks, top5 replication, single-checkpoint PatchTST, zero-mean EMA, Schedule-Free SGD sweep, noise onset и 90 epoch OOD horizon.

### Paired Robustness Summary

Положительная delta означает, что learnable method бьет matched cosine в этой perturbation cell.

| Experiment | Perturbation | Method | Cosine | Method | Delta |
|---|---|---|---:|---:|---:|
| shock | shock_lr=0.10 | patchtst | 69.45 | **70.17** | **+0.72** |
| shock | shock_lr=0.10 | residgru | 69.45 | 69.31 | -0.14 |
| noshock | noshock | patchtst | 69.90 | 69.88 | -0.02 |
| O6_long90 | long90 | patchtst | 70.10 | 69.99 | -0.11 |
| O5_noise_onset_e15 | label=0.20@e15 | patchtst | 64.50 | 64.25 | -0.25 |
| innoise | input_noise=0.05 | patchtst | 65.98 | 65.90 | -0.08 |
| innoise | input_noise=0.15 | patchtst | 57.33 | 56.99 | -0.34 |
| lblnoise | label_noise=0.10 | patchtst | 66.60 | 66.34 | -0.26 |
| lblnoise | label_noise=0.20 | patchtst | 63.71 | 63.19 | -0.52 |
| lblnoise | label_noise=0.30 | patchtst | 60.81 | 60.29 | -0.52 |
| compound | label=0.20+shock_lr=1.0@e15 | patchtst | 64.08 | 63.11 | -0.98 |
| shock | shock_lr=0.50 | patchtst | 70.02 | 69.53 | -0.49 |
| shock | shock_lr=1.00 | patchtst | 69.90 | 69.69 | -0.21 |
| shock | shock_lr=2.00 | patchtst | 62.14 | 59.77 | -2.37 |
| shock | shock_lr=1.00 | residgru | 69.90 | 62.83 | -7.07 |
| shock | shock_lr=2.00 | residgru | 62.14 | 60.19 | -1.95 |

Промежуточный вывод: positive region узкий. PatchTST помогает, когда LR временно слишком маленький, но не когда доминирует noise objective или LR принудительно слишком высокий.

## Winning Seeds Case

Самый воспроизводимый positive case - mild LR-down shock. На epochs 20-21 student LR принудительно ставится в `0.10`, затем scheduler resumes. PatchTST выигрывает 3 из 4 paired seeds.

| Seed | Cosine final acc | PatchTST final acc | Delta | Status |
|---:|---:|---:|---:|---|
| 1 | 69.94 | 69.51 | -0.43 | non-win |
| 2 | 69.58 | 70.30 | +0.72 | WIN |
| 3 | 69.71 | 70.44 | +0.73 | WIN |
| 4 | 68.57 | 70.42 | +1.85 | WIN |

Финальные student и teacher checkpoints для всех paired shock01 runs лежат в `checkpoints/winning_shock01/`.

Промежуточный вывод: это лучший кейс для отдельной qualitative демонстрации. Он paired by seed, имеет positive mean delta и понятный механизм: cosine привязан к epoch index, а residual policy может корректировать траекторию после down-shock.

## Noise and Compound Perturbations

| Perturbation | Cosine | PatchTST | Delta | Вывод |
|---|---:|---:|---:|---|
| input_noise=0.05 | 65.98 +/- 0.43 | 65.90 +/- 0.44 | -0.08 | tie |
| input_noise=0.15 | 57.33 +/- 0.37 | 56.99 +/- 0.11 | -0.34 | slight cosine win |
| label_noise=0.10 | 66.60 +/- 0.49 | 66.34 +/- 0.23 | -0.26 | tie |
| label_noise=0.20 | 63.71 +/- 0.23 | 63.19 +/- 0.62 | -0.52 | cosine win |
| label_noise=0.30 | 60.81 +/- 0.28 | 60.29 +/- 0.34 | -0.52 | cosine win |
| label=0.20 + shock_lr=1.0@e15 | 64.08 +/- 0.35 | 63.11 +/- 0.44 | -0.98 | cosine win |
| label=0.20@e15 onset | 64.50 +/- 0.10 | 64.25 +/- 0.38 | -0.25 | tie |

Промежуточный вывод: policy corrections не решают label/input corruption. При noisy data cosine остается очень сильным regularized baseline.

## OOD Horizon and Replication Checks

| Open item | n | Result | Delta / interpretation |
|---|---:|---:|---|
| PatchTST top5 blend075 replication | 6 | 69.99 +/- 0.44 | -0.21 vs 60ep cosine ref |
| Single-checkpoint PatchTST | 5 | 69.82 +/- 0.29 | хуже top5 ensemble |
| Zero-mean EMA ensemble fix | 3 | 69.72 +/- 0.31 | не помогло |
| 90ep OOD horizon, cosine | 5 | 70.10 +/- 0.40 | reference |
| 90ep OOD horizon, PatchTST | 5 | 69.99 +/- 0.10 | -0.11, much lower variance |

Промежуточный вывод: 70.70 headline вероятнее high seed, а не новое среднее. Но OOD horizon stability остается полезным свойством: PatchTST имеет меньшую variance на 90 эпохах.

## External Adaptive-LR Comparators

Ни одна из внешних adaptive-LR controller families не дошла до cosine regime в том же 60 epoch LoT distillation setting.

| Method family | Variant | n | Final mean | Final std | Delta vs cosine mean |
|---|---|---:|---:|---:|---:|
| AdaLRS | default | 3 | 67.41 | 0.82 | -2.79 |
| AdaLRS | aggressive | 3 | 65.99 | 2.29 | -4.21 |
| AdaLRS | narrow_safe | 3 | 54.73 | 4.92 | -15.46 |
| Bandit EXP3 | safe | 3 | 53.61 | 17.94 | -16.58 |
| Bandit EXP3 | fast | 3 | 52.09 | 6.49 | -18.10 |
| Bandit UCB | ucb | 3 | 40.88 | 23.43 | -29.31 |
| Hypergrad-HB | fast | 3 | 42.67 | 8.21 | -27.53 |
| Hypergrad-HB | safe | 3 | 39.75 | 3.95 | -30.45 |
| Hypergrad-HB | smooth | 3 | 34.09 | 8.06 | -36.11 |

Schedule-Free SGD LR sweep from robust12h:

| SF-SGD LR | n | Final mean | Final std | Delta vs 35ep cosine ref |
|---:|---:|---:|---:|---:|
| 0.1 | 3 | 39.54 | 1.42 | -30.27 |
| 0.3 | 3 | 49.02 | 1.13 | -20.79 |
| 0.7 | 3 | 57.79 | 0.38 | -12.02 |
| 1.0 | 3 | 58.93 | 1.39 | -10.88 |

Промежуточный вывод: negative controls важны. Результат не в том, что "любой adaptive LR помогает"; полезное поведение специфично для bounded residual over cosine.

## Improve5h Postmortem

Follow-up sweep по blends, warmups, cosine handoff и ensemble sizes не нашел конфигурацию лучше:

- Best improve5h run: `outens_blend0800_warm05_seed3_60ep` at **70.29**, ниже cosine seed3 **70.45** и ниже previous PatchTST headline **70.70**.
- Root cause: output ensemble predictions схлопнулись почти в constant bias.
- Per-epoch `policy_pred` standard deviation на epochs 10-55 был только около `0.003`, тогда как mean bias был около `-0.264` для top5.
- Sweep в основном менял fixed cosine attenuation в диапазоне около `0.94x` to `0.98x`, а не state-dependent control.

Промежуточный вывод: дальнейший tuning blend/EMA/warmup вряд ли даст robust mean improvement. Более полезные направления требуют новых state-dependent targets или perturbation-aware training.

## Final Interpretation

Данные поддерживают такой осторожный claim:

1. Bounded residual-log LR policy может безопасно match cosine на normal и OOD horizons.
2. PatchTST - лучший compact backbone для policy.
3. Самый сильный positive result - trajectory repair после mild LR-down perturbation.
4. Подход не стабильно бьет cosine на label noise, input noise, compound perturbations или extreme LR shocks.
5. Single-seed 70.70 headline реален и воспроизводим из bundled checkpoint/logs, но multi-seed mean tied with cosine rather than better.

## Состав пакета

- `code/` - package-local code snapshot для reproduction scripts.
- `data/oracle/` - oracle features и targets для обучения LR-policy.
- `data/robust12h/` - parsed robustness tables: `cells.csv`, `pairs.csv`, `per_run.csv`, `per_epoch.csv`.
- `data/cifar/cifar-100-python/` - bundled CIFAR-100 cache.
- `checkpoints/policies_beatcos/` - пять PatchTST policy checkpoints для top-5 ensemble.
- `checkpoints/final_headline_seed3/` - final student/teacher checkpoints для 70.70 seed-3 run.
- `checkpoints/winning_shock01/` - final student/teacher checkpoints для paired cosine и PatchTST shock01 seeds 1-4.
- `logs/` и `snapshots/` - raw logs и per-epoch metrics для аудита таблиц.
- `figures/` - headline и robust12h figures.
- `reports/` - curated source reports плюс `reports/source_results/`, сохраненная копия всех markdown result reports из project `results/` tree.
- `sota_comparison/` - archived negative-control adaptive LR comparator results.

## Воспроизведение

Из корня пакета на CUDA machine:

```bash
PYTHON_BIN=/path/to/python bash scripts/reproduce_headline_seed3.sh
```

Чтобы воспроизвести paired LR-down shock case:

```bash
PYTHON_BIN=/path/to/python bash scripts/reproduce_shock01_winning_seeds.sh
```

Чтобы пересобрать package-local tables/summary и figures из bundled CSV/logs:

```bash
python scripts/analyze_robust12h.py
python scripts/plot_figures.py
```

## Data Notes

CIFAR-100 bundled under `data/cifar/cifar-100-python/`. Reproduction scripts still pass `--download` to torchvision; это harmless, когда cache уже есть, и дает fallback, если директорию удалили. Все experiment-specific data для LR-policy - oracle arrays, policy checkpoints, logs, parsed tables - включены в пакет.

## Integrity

`MANIFEST.sha256` содержит SHA256 hashes всех файлов пакета, кроме самого manifest. Проверка:

```bash
sha256sum -c MANIFEST.sha256
```
