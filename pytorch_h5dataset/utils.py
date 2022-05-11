from torch.nn import Module

class NormImageUint8ToFloat(Module):
    def forward(self, im) :
        return  2.*((im/255.)-.5)