# add these lines in your train.py
import colossalai

args = colossalai.get_default_parser().parse_args()

colossalai.launch(config='./config.py',
                  rank=1,
                  world_size=1,
                  host=1,
                  port=1,
                  backend='nccl'
)

breakpoint()