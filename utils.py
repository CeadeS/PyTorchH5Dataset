from torch.nn import Module

class NormImage(Module):
    def forward(self, im) :
        return  2.*((im/255.)-.5)