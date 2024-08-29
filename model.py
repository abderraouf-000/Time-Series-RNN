import torch
import torch.nn as nn




class Rnn(nn.Module):
    def __init__(self,input_size , hidden_size , output_size):
        super(Rnn,self).__init__();
        self.hidden_size = hidden_size;
        self.input_size = input_size;
        self.output_size = output_size;
        self.i2h = nn.Linear(self.input_size + self.hidden_size , self.hidden_size);
        self.i2o = nn.Linear(self.input_size + self.hidden_size , self.output_size);
        self.softmax = nn.LogSoftmax(dim = 1);
    
    def forward(self,input_tensor,hidden_tensor):
        combined = torch.cat((input_tensor , hidden_tensor) , 1);
        hidden = self.i2h(combined)
        output = self.softmax(self.i2o(combined));
        return output,hidden;

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size);
    

rnn = Rnn(20 , 128 , 1)
hidden_tensor = rnn.init_hidden();
output, next_hidden = rnn(torch.zeros(1,20) , hidden_tensor);
print(output,next_hidden)
