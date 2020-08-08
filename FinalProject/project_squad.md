
1. Tensorboard
    在Azure上训练并在本地显示：
    1. 在VM终端运行`tensorboard --logdir save --port 5678 # Start TensorBoard`
    2. 打开本地终端，运行`ssh -N -f -L localhost:1234:localhost:5678 hy2632@13.90.229.131`
    3. 本地打开`localhost:1234`

