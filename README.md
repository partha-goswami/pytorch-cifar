## pytorch-cifar
The repository contains various experiments on cifar10 and cifar100 datasets using pytorch.

## Methods used in utils.py

| Method                             | Purpose                                                            |
| ---------------------------------- |:------------------------------------------------------------------:|
| get_device                         | returns current used device information (gpu/cuda or cpu)          |
| get_config_values                  | defines config dictionary                                          |
| get_cifar10_stats                  | returns cifar10 data stats like mean, std                          |
| apply_albumentation                | applies albumentation transforms                                   |
| get_data_loaders                   | returns data loaders                                               |
| get_optimizer                      | returns optimizer                                                  |
| get_scheduler                      | returns scheduler                                                  |
| identify_wrong_predictions         | identifies wrong predictions                                       |
| plot_metrics                       | plots accuracy and loss progression                                |
| plot_gradCAM                       | plots gradCAM data                                                 |

## Methods used in main.py

| Method                             | Purpose                                                            |
| ---------------------------------- |:------------------------------------------------------------------:|
| train                              | trains the model                                                   |
| test                               | tests the model                                                    |
| print_model_summary                | returns model summary                                              |
| experiment                         | runs experiment                                                    |

## Best Test Accuracy Obtained

| Model                              | Accuracy                                                            |
| ---------------------------------- |:------------------------------------------------------------------:|
| ResNet18 (20 epochs)               | 92.00                                                  |
