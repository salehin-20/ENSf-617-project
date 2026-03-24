# Model Comparison

| Model | MAE | RMSE | MAPE | Pinball | Notes |
| --- | --- | --- | --- | --- | --- |
| LSTM baseline | 0.279 | 0.422 | 2.26% | 0.091 | normalized units; horizon 24 |
| TFT (15 epochs) | 834.781 | 1286.845 | 6.18% | 584.738 | horizon 24, lookback 336; metrics from reports/tft/metrics.yaml |
