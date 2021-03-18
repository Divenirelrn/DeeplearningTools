class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.decay) * x + self.decay * self.shadow[name]
        self.shadow[name] = new_average.clone()


#使用方法，分为初始化、注册和更新三个步骤。
#init
ema = EMA(0.999)

#register
for name, param in model.named_parameters():
    if param.requires_grad:
        ema.register(name, param.data)

#update
for name, param in model.named_parameters():
    if param.requires_grad:
        ema.update(name, param.data)