from colossalai.shardformer.policies.base_policy import Policy

class MyPolicy(Policy):
    # implement your own policy
    ...

# init model and shard former
...

# use customized policy to shard model
my_policy = MyPolicy()
shard_former.optimize(model, my_policy)


