# NNResiliency
Analog Deep Neural Network Simulation

# Examples

- Train a model on the device 3
  ```
  python main.py --net_type mlp --dataset morse --training_noise 0.05 --training_noise_mean 0 --device 3 --testing_noise 0 --testing_noise_mean 0 --batch_size 200 -a 120 --num_epochs 600
  ```
- Test the model trained with specified parameters on the device 3.
  ```
  python main.py --net_type mlp --dataset morse --training_noise 0.05 --training_noise_mean 0 --device 3 --testing_noise 0 0.02 0.04 0.08 0.1 0.12 0.14 0.16 0.18 0.2 --testing_noise_mean 0 --batch_size 200 -a 120 --num_epochs 600 --testOnly
  ```
  The results will be saved as json files under `test/`
