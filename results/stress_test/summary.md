# Stress Test Results

These are availability and robustness stress tests for guideline scores under sparse observations. Missing coverage is a result, not a low-risk prediction.

## Coverage Decay

| Baseline | Retention | N | Covered | Coverage | Mean score points | Complete rate | Mean final components |
|---|---:|---:|---:|---:|---:|---:|---:|
| qsofa | 1.0 | 599 | 59 | 0.0985 | 1.1937 | 1.0000 | 3.0000 |
| qsofa | 0.75 | 599 | 58 | 0.0968 | 0.9098 | 1.0000 | 3.0000 |
| qsofa | 0.5 | 599 | 56 | 0.0935 | 0.6227 | 1.0000 | 3.0000 |
| qsofa | 0.25 | 599 | 55 | 0.0918 | 0.3372 | 1.0000 | 3.0000 |
| qsofa | 0.1 | 599 | 47 | 0.0785 | 0.1720 | 1.0000 | 3.0000 |
| sirs | 1.0 | 599 | 51 | 0.0851 | 0.1002 | 1.0000 | 4.0000 |
| sirs | 0.75 | 599 | 38 | 0.0634 | 0.0735 | 1.0000 | 4.0000 |
| sirs | 0.5 | 599 | 26 | 0.0434 | 0.0518 | 1.0000 | 4.0000 |
| sirs | 0.25 | 599 | 17 | 0.0284 | 0.0351 | 1.0000 | 4.0000 |
| sirs | 0.1 | 599 | 9 | 0.0150 | 0.0167 | 1.0000 | 4.0000 |
| sofa | 1.0 | 599 | 599 | 1.0000 | 38.5776 | 0.0000 | 1.2404 |
| sofa | 0.75 | 599 | 599 | 1.0000 | 29.1753 | 0.0000 | 1.2421 |
| sofa | 0.5 | 599 | 599 | 1.0000 | 19.7596 | 0.0000 | 1.2404 |
| sofa | 0.25 | 599 | 596 | 0.9950 | 10.5876 | 0.0000 | 1.2500 |
| sofa | 0.1 | 599 | 584 | 0.9750 | 4.8164 | 0.0000 | 1.2551 |

## Component Availability

| Baseline | Retention | Component | Patient fraction any | Mean step fraction |
|---|---:|---|---:|---:|
| qsofa | 1.0 | RespRate | 0.2738 | 0.2419 |
| qsofa | 1.0 | SysABP | 0.6795 | 0.4870 |
| qsofa | 1.0 | GCS | 0.9783 | 0.2649 |
| qsofa | 0.75 | RespRate | 0.2738 | 0.2421 |
| qsofa | 0.75 | SysABP | 0.6795 | 0.4858 |
| qsofa | 0.75 | GCS | 0.9783 | 0.2654 |
| qsofa | 0.5 | RespRate | 0.2738 | 0.2424 |
| qsofa | 0.5 | SysABP | 0.6761 | 0.4826 |
| qsofa | 0.5 | GCS | 0.9783 | 0.2651 |
| qsofa | 0.25 | RespRate | 0.2738 | 0.2424 |
| qsofa | 0.25 | SysABP | 0.6745 | 0.4699 |
| qsofa | 0.25 | GCS | 0.9616 | 0.2709 |
| qsofa | 0.1 | RespRate | 0.2721 | 0.2375 |
| qsofa | 0.1 | SysABP | 0.6611 | 0.4380 |
| qsofa | 0.1 | GCS | 0.8581 | 0.2796 |
| sirs | 1.0 | Temp | 0.9783 | 0.3208 |
| sirs | 1.0 | HR | 0.9783 | 0.8565 |
| sirs | 1.0 | RespRate | 0.2738 | 0.2419 |
| sirs | 1.0 | WBC | 0.9850 | 0.0643 |
| sirs | 0.75 | Temp | 0.9783 | 0.3198 |
| sirs | 0.75 | HR | 0.9783 | 0.8552 |
| sirs | 0.75 | RespRate | 0.2738 | 0.2421 |
| sirs | 0.75 | WBC | 0.9499 | 0.0632 |
| sirs | 0.5 | Temp | 0.9783 | 0.3211 |
| sirs | 0.5 | HR | 0.9783 | 0.8542 |
| sirs | 0.5 | RespRate | 0.2738 | 0.2424 |
| sirs | 0.5 | WBC | 0.8447 | 0.0622 |
| sirs | 0.25 | Temp | 0.9683 | 0.3290 |
| sirs | 0.25 | HR | 0.9783 | 0.8471 |
| sirs | 0.25 | RespRate | 0.2738 | 0.2424 |
| sirs | 0.25 | WBC | 0.6177 | 0.0623 |
| sirs | 0.1 | Temp | 0.8965 | 0.3313 |
| sirs | 0.1 | HR | 0.9750 | 0.8263 |
| sirs | 0.1 | RespRate | 0.2721 | 0.2375 |
| sirs | 0.1 | WBC | 0.3639 | 0.0653 |
| sofa | 1.0 | Platelets | 0.9850 | 0.0700 |
| sofa | 1.0 | Bilirubin | 0.4441 | 0.0154 |
| sofa | 1.0 | Creatinine | 0.9866 | 0.0748 |
| sofa | 1.0 | GCS | 0.9783 | 0.2649 |
| sofa | 1.0 | MAP | 0.6728 | 0.4823 |
| sofa | 1.0 | PaO2/FiO2 | 0.4274 | 0.0141 |
| sofa | 0.75 | Platelets | 0.9566 | 0.0695 |
| sofa | 0.75 | Bilirubin | 0.3656 | 0.0147 |
| sofa | 0.75 | Creatinine | 0.9616 | 0.0742 |
| sofa | 0.75 | GCS | 0.9783 | 0.2654 |
| sofa | 0.75 | MAP | 0.6728 | 0.4810 |
| sofa | 0.75 | PaO2/FiO2 | 0.3756 | 0.0145 |
| sofa | 0.5 | Platelets | 0.8648 | 0.0680 |
| sofa | 0.5 | Bilirubin | 0.2755 | 0.0142 |
| sofa | 0.5 | Creatinine | 0.8781 | 0.0737 |
| sofa | 0.5 | GCS | 0.9783 | 0.2651 |
| sofa | 0.5 | MAP | 0.6711 | 0.4776 |
| sofa | 0.5 | PaO2/FiO2 | 0.2922 | 0.0141 |
| sofa | 0.25 | Platelets | 0.6411 | 0.0684 |
| sofa | 0.25 | Bilirubin | 0.1836 | 0.0146 |
| sofa | 0.25 | Creatinine | 0.6477 | 0.0715 |
| sofa | 0.25 | GCS | 0.9616 | 0.2709 |
| sofa | 0.25 | MAP | 0.6711 | 0.4644 |
| sofa | 0.25 | PaO2/FiO2 | 0.1686 | 0.0127 |
| sofa | 0.1 | Platelets | 0.3873 | 0.0710 |
| sofa | 0.1 | Bilirubin | 0.0968 | 0.0147 |
| sofa | 0.1 | Creatinine | 0.3689 | 0.0724 |
| sofa | 0.1 | GCS | 0.8581 | 0.2796 |
| sofa | 0.1 | MAP | 0.6561 | 0.4322 |
| sofa | 0.1 | PaO2/FiO2 | 0.0902 | 0.0125 |

## Covered-Patient Thinning Sensitivity

| Baseline | Retention | Ref covered | Thin covered | Pairs | Paired coverage | Lost coverage | Mean prob drift | Mean score drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qsofa | 1.0 | 59 | 59 | 59 | 0.0985 | 0 | 0.0000 | 0.0000 |
| qsofa | 0.75 | 59 | 58 | 58 | 0.0968 | 1 | 0.0000 | 0.0000 |
| qsofa | 0.5 | 59 | 56 | 56 | 0.0935 | 3 | 0.0000 | 0.0000 |
| qsofa | 0.25 | 59 | 55 | 55 | 0.0918 | 4 | 0.0000 | 0.0000 |
| qsofa | 0.1 | 59 | 47 | 47 | 0.0785 | 12 | 0.0000 | 0.0000 |
| sirs | 1.0 | 51 | 51 | 51 | 0.0851 | 0 | 0.0000 | 0.0000 |
| sirs | 0.75 | 51 | 38 | 38 | 0.0634 | 13 | 0.0000 | 0.0000 |
| sirs | 0.5 | 51 | 26 | 26 | 0.0434 | 25 | 0.0000 | 0.0000 |
| sirs | 0.25 | 51 | 17 | 17 | 0.0284 | 34 | 0.0000 | 0.0000 |
| sirs | 0.1 | 51 | 9 | 9 | 0.0150 | 42 | 0.0000 | 0.0000 |
| sofa | 1.0 | 599 | 599 | 599 | 1.0000 | 0 | 0.0000 | 0.0000 |
| sofa | 0.75 | 599 | 599 | 599 | 1.0000 | 0 | 0.0027 | 0.0634 |
| sofa | 0.5 | 599 | 599 | 599 | 1.0000 | 0 | 0.0069 | 0.1619 |
| sofa | 0.25 | 599 | 596 | 596 | 0.9950 | 3 | 0.0100 | 0.2399 |
| sofa | 0.1 | 599 | 584 | 584 | 0.9750 | 15 | 0.0128 | 0.3305 |

## Repeated-Thinning Availability

| Baseline | Retention | N patients | Scorable thinned versions | Patients scorable in some repeats but not others | Mean SD of predicted risk across scorable repeats |
|---|---:|---:|---:|---:|---:|
| qsofa | 0.75 | 599 | 9.6494% | 6 / 599 (1.0017%) | 0.0000 |
| qsofa | 0.5 | 599 | 9.3823% | 8 / 599 (1.3356%) | 0.0000 |
| qsofa | 0.25 | 599 | 8.8815% | 17 / 599 (2.8381%) | 0.0000 |
| qsofa | 0.1 | 599 | 7.3790% | 35 / 599 (5.8431%) | 0.0000 |
| sirs | 0.75 | 599 | 6.5776% | 40 / 599 (6.6778%) | 0.0000 |
| sirs | 0.5 | 599 | 4.5409% | 47 / 599 (7.8464%) | 0.0000 |
| sirs | 0.25 | 599 | 2.5209% | 43 / 599 (7.1786%) | 0.0000 |
| sirs | 0.1 | 599 | 1.3856% | 29 / 599 (4.8414%) | 0.0000 |
| sofa | 0.75 | 599 | 100.0000% | 0 / 599 (0.0000%) | 0.0070 |
| sofa | 0.5 | 599 | 99.9165% | 2 / 599 (0.3339%) | 0.0102 |
| sofa | 0.25 | 599 | 99.6661% | 9 / 599 (1.5025%) | 0.0124 |
| sofa | 0.1 | 599 | 98.3639% | 41 / 599 (6.8447%) | 0.0141 |

## Prefix Availability and Volatility

| Baseline | N patients | Mean prefix coverage | Mean first covered prefix | Mean TV | Mean max jump | Mean unavailable-after-available |
|---|---:|---:|---:|---:|---:|---:|
| qsofa | 599 | 0.0873 | 7.9153 | 0.0000 | 0.0000 | 0.0000 |
| sirs | 599 | 0.0549 | 19.9804 | 0.0000 | 0.0000 | 0.0000 |
| sofa | 599 | 0.9910 | 2.4758 | 1.0723 | 0.1317 | 0.0000 |

## Covered-Subset Performance Under Sparsity

| Baseline | Retention | Covered N | Coverage | AUC | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| qsofa | 1.0 | 59 | 0.0985 | 0.5000 | 0.2500 | 0.6931 | 0.3814 |
| qsofa | 0.75 | 58 | 0.0968 | 0.5000 | 0.2500 | 0.6931 | 0.3793 |
| qsofa | 0.5 | 56 | 0.0935 | 0.5000 | 0.2500 | 0.6931 | 0.3929 |
| qsofa | 0.25 | 55 | 0.0918 | 0.5000 | 0.2500 | 0.6931 | 0.3909 |
| qsofa | 0.1 | 47 | 0.0785 | 0.5000 | 0.2500 | 0.6931 | 0.3723 |
| sirs | 1.0 | 51 | 0.0851 | 0.5000 | 0.2465 | 0.6861 | 0.3578 |
| sirs | 0.75 | 38 | 0.0634 | 0.5000 | 0.2467 | 0.6865 | 0.3372 |
| sirs | 0.5 | 26 | 0.0434 | 0.5000 | 0.2466 | 0.6864 | 0.3412 |
| sirs | 0.25 | 17 | 0.0284 | 0.5000 | 0.2468 | 0.6868 | 0.3186 |
| sirs | 0.1 | 9 | 0.0150 | 0.5000 | 0.2473 | 0.6877 | 0.2729 |
| sofa | 1.0 | 599 | 1.0000 | 0.5370 | 0.2484 | 0.6901 | 0.3596 |
| sofa | 0.75 | 599 | 1.0000 | 0.5285 | 0.2494 | 0.6919 | 0.3605 |
| sofa | 0.5 | 599 | 1.0000 | 0.5379 | 0.2500 | 0.6932 | 0.3617 |
| sofa | 0.25 | 596 | 0.9950 | 0.5427 | 0.2511 | 0.6955 | 0.3643 |
| sofa | 0.1 | 584 | 0.9750 | 0.5374 | 0.2525 | 0.6984 | 0.3661 |

## Timestamp Stretching

Not applicable in v1: guideline scores do not directly model elapsed-time dynamics.
