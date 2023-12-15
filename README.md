# PPO-simplest
PPO in one python file. It is the simplest implementation of PPO, which is easy to understand and modify. The code is based on [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), but remove complex dependencies and make it more concise. I also
add Pong example which learns from pixels. 

The only dependency is PyTorch and gymnasium(gym):
```bash
pip install "gymnasium[atari, accept-rom-license, classic-control]"
```

## CartPole
```
python ppo.py --env CartPole-v1 --steps 500 --kl 0.01 --device cpu
```

Initial statistics:
```
Number of parameters:    pi: 1794,       v: 1537
Epoch: 0
{'EpRet': 36.0, 'EpLen': 36}
{'EpRet': 29.0, 'EpLen': 29}
{'EpRet': 27.0, 'EpLen': 27}
{'EpRet': 31.0, 'EpLen': 31}
{'EpRet': 21.0, 'EpLen': 21}
{'EpRet': 15.0, 'EpLen': 15}
{'EpRet': 76.0, 'EpLen': 76}
{'EpRet': 85.0, 'EpLen': 85}
{'EpRet': 40.0, 'EpLen': 40}
{'EpRet': 18.0, 'EpLen': 18}
{'EpRet': 33.0, 'EpLen': 33}
{'EpRet': 28.0, 'EpLen': 28}
{'EpRet': 31.0, 'EpLen': 31}
{'EpRet': 26.0, 'EpLen': 26}
{'EpRet': 4.0, 'EpLen': 4}
{'LossPi': -0.009821644984185696, 'LossV': 579.6947631835938, 'KL': 0.0012354411883279681, 'Entropy': 0.6754535436630249, 'ClipFrac': 0.00800000037997961}
```

After several epochs, it can totally master this game (just minutes on CPU, you can run more time to make the entropy lower):
```
Epoch: 1240
{'EpRet': 500.0, 'EpLen': 500}
{'LossPi': -0.0007033138535916805, 'LossV': 0.024762842804193497, 'KL': 0.003935517277568579, 'Entropy': 0.2533852458000183, 'ClipFrac': 0.004000000189989805}
```
To see the real-time graphics, you can run:
```
python ppo.py --env CartPole-v1 --steps 5000 --kl 0.01 --device cpu --render --load_from
```
<p align="center">
  <img src="CartPole.png" width="589" height="424" alt="Cute Llama">
</p>

## Pong
A more interesting example is Pong, which learns from pixels. You'd better run it on GPU to speed up the training. 
```
python ppo.py --from_pixel
```
After several hours, you can see that `EpRet` becomes positive, which means it can beat the built-in AI. If you want a more powerful model, you can make the CNN bigger and tune the hyper-parameters. 
<p align="center">
  <img src="Pong.png" width="408" height="512" alt="Cute Llama">
</p>

## Valuable Links
- Vanilla Policy Gradients(actor-critic) based on numpy (mannually compute gradients): [link](https://github.com/schinger/pong_actor-critic)
- PPO on LLM (Llama2): [link](https://github.com/schinger/FullLLM)
