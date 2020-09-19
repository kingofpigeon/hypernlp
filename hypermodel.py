import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
class HyberRnnNet(nn.Module):
    def __init__(self, 
    inputs_dim:int, 
    hidden_hat_dim:int, 
    layer_nums:int, 
    rnn_type:str,
    bidirectional =False):

        super(HyberRnnNet, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_hat_dim = hidden_hat_dim
        self.layer_nums = layer_nums
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        
        if self.rnn_type == 'rnn':
            self.RNN = nn.RNN(input_size = self.inputs_dim, 
            hidden_size = self.hidden_hat_dim,
            num_layers = self.layer_nums,
            batch_first = True,
            dropout = 0.2,
            bidirectional = bidirectional
                            )

        elif self.rnn_type == 'gru':
            self.RNN = nn.GRU(input_size = self.inputs_dim, 
            hidden_size = self.hidden_hat_dim,
            num_layers = self.layer_nums,
            batch_first = True,
            dropout = 0.2,
            bidirectional = bidirectional
                             )

        elif self.rnn_type == 'lstm':
            self.RNN = nn.LSTM(input_size = self.inputs_dim, 
            hidden_size = self.hidden_hat_dim,
            num_layers = self.layer_nums,
            batch_first = True,
            dropout = 0.2,
            bidirectional = bidirectional
                              )
            #torch.nn.init.orthogonal_()

        else:
            raise ValueError(
        "rnn_type should be rnn, gru or lstm!"
        )

    def forward(self,inputs, state_0): # inputs:(batch, seq_len, input_size) ; h_0:(num_layers * num_directions, batch, hidden_size)
        if self.rnn_type == 'lstm':
            (h_0, c_0) = state_0
            output, (h_t, c_t) = self.RNN(inputs, (h_0, c_0))
            return output, (h_t, c_t)
        
        else:
            output, h_t = self.RNN(inputs, state_0) #h_t: h^{hat}_{t}
            return output, h_t

    def init_state(self, inputs):
        batch_size = inputs.size(0)
        a= 2 if self.bidirectional else 1
        if self.rnn_type == 'lstm':
            h_0 = torch.randn(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            c_0 = torch.randn(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            return (h_0, c_0)

        else:
            h_0 = torch.zeros(self.layer_nums*a, batch_size, self.hidden_hat_dim, device=inputs.device)
            return h_0
        
        
class InferenceNetRnnCell(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int, 
    activate = 'tanh'
   ):
        super(InferenceNetRnnCell, self).__init__()
        self.w_hh = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hx = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hb = nn.Linear(hidden_hat_dim, z_dim, bias=True)
        self.W_hz = nn.Linear(z_dim, hidden_dim,bias=False)
        self.W_xz = nn.Linear(z_dim, hidden_dim, bias=False)
        self.b = nn.Linear(z_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.Wx = nn.Linear(input_dim ,hidden_dim)
        self.activate = activate
        self.dropout = nn.Dropout(p=0.1, inplace=True)
       
    def forward(self, h_t, h_t_hat, inf_inputs):

        z_h = self.w_hh(h_t_hat) #z_{h}
        z_x = self.w_hx(h_t_hat) # z_{x}
        z_bias = self.w_hb(h_t_hat) #z_{b}
        d_z_h = self.W_hz(z_h) #d_{h}(z_{h})
        d_z_x = self.W_xz(z_x) #d_{x}(z_{x})
        b_z_b = self.b(z_bias) #d_{b}{z_{b}}
        h_t_new = d_z_h*self.Wh(h_t)+d_z_x*self.Wx(inf_inputs)+ b_z_b
        h_t_new = self.dropout(h_t_new)
        
        if self.activate =='relu':
            return torch.relu(h_t_new)
        elif self.activate =='tanh':
            return torch.tanh(h_t_new)
        elif self.activate =='sigmoid':
            return torch.sigmoid(h_t_new)
    
class InferenceNetLSTMCell(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int
   ):

        super(InferenceNetLSTMCell, self).__init__()
        self.w_hh = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hx = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hb = nn.Linear(hidden_hat_dim, z_dim)
        self.W_hz = nn.Linear(z_dim, 4*hidden_dim,bias=False)
        self.W_xz = nn.Linear(z_dim, 4*hidden_dim, bias=False)
        self.b = nn.Linear(z_dim, 4*hidden_dim)
        self.Wh = nn.Linear(hidden_dim, 4*hidden_dim)
        self.Wx = nn.Linear(input_dim ,4*hidden_dim)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_c = nn.LayerNorm(hidden_dim)
    def forward(self, h_t, c, h_t_hat, inf_inputs):

        z_h = self.w_hh(h_t_hat) #z_{h}   size = (b, z_dim)
        z_x = self.w_hx(h_t_hat) #z_{x}  size = (b,z_dim)
        z_bias = self.w_hb(h_t_hat) # z_{b}  size = (b, z_dim)
        d_z_h = self.W_hz(z_h) #d_{h}(z_{h}) size = (b, 4*hidden_dim)
        d_z_x = self.W_xz(z_x) #d_{x}(z_{x})  size = (b, 4*hidden_dim)
        b_z_b = self.b(z_bias) #d_{b}{z_{b}}  size = (b, 4*hidden_dim)
        ifgo = d_z_h*self.Wh(h_t)+d_z_x*self.Wx(inf_inputs)+ b_z_b    #size = (b, 4*hidden_dim)
        i,f,g,o = torch.chunk(ifgo,4,-1)  #i ,f,g,o, size = (b, hidden_dim)
        i = torch.sigmoid(i) 
        f = torch.sigmoid(f)
        g = torch.sigmoid(g)
        o = torch.sigmoid(o)
        new_c =f*c+i*c
        new_h = o*torch.tanh(new_c)
        new_h  = self.dropout(new_h)
        new_h = self.norm_h(new_h)
        new_c = self.norm_c(new_c)
        return new_h, new_c

class InferenceRNN(nn.Module):
    def __init__(self, 
    z_dim:int, 
    input_dim:int,
    hidden_hat_dim:int,
    hidden_dim:int,
    cell_type: str
     ):
        super(InferenceRNN, self).__init__()
        self.cell_type = cell_type
        if cell_type =='rnn':
            self.cell = InferenceNetRnnCell(z_dim, input_dim, hidden_hat_dim, hidden_dim)
        elif cell_type =='lstm':
            self.cell = InferenceNetLSTMCell(z_dim, input_dim, hidden_hat_dim, hidden_dim)
        else:
            raise ValueError("need gru?")
            
    def forward(self, state, h_hat_t, inf_inputs):
        outputs = []
        if self.cell_type =='lstm':
            h,c = state
            for t in range(inf_inputs.size(0)):
                h,c = self.cell(h,c, h_hat_t, inf_inputs[t])
                outputs.append(h)
        else:
            h = state
            for t in range(inf_inputs.size(0)):
                h = self.cell(h, h_hat_t, inf_inputs[t])
                outputs.append(h)
        
        return torch.stack(outputs,1)
from transformers import AlbertForQuestionAnswering, AlbertModel
    
class HyperQuestionAnswering(AlbertForQuestionAnswering):
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(int(config.hidden_size/4), config.num_labels)
        self.hypernet = HyberRnnNet(2048, 512,1, 'lstm')
        self.infernet = InferenceRNN(64, 2048,  512, 512, 'lstm')
        

   
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
        title = None,
        t_mask = None,
        t_lens = None,
        c_lens = None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        with torch.no_grad():
            
            hyper_inputs = self.albert(
            input_ids = title,
            attention_mask=t_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
            
            infer_inputs = self.albert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        
        state = self.hypernet.init_state(hyper_inputs[0])
        title_len = t_lens
        content_len = c_lens
        check = torch.isnan(hyper_inputs[0])
        if True in check:
            print('nan in hyper_inputs!')
        check = torch.isnan(infer_inputs[0])
        if True in check:
            print('nan in infer_inputs!')
        outputs, state = self.hypernet(hyper_inputs[0], state)
        check = torch.isnan(outputs)
        if True in check:
            print('nan in outputs!')
        h_hat_t = torch.stack([t[l-1] for (t,l) in zip(outputs,title_len)])
    
        if isinstance(state, tuple):
            state = list(state)
            state[0] =state[0][-1]
            state[1] =state[1][-1]
        else:
            state = state[-1]
        infer_inputs_ = infer_inputs[0].transpose(0,1).contiguous()
        infer_outputs = self.infernet(state, h_hat_t, infer_inputs_)
        
        
        check = torch.isnan(infer_outputs)
        if True in check:
            print('nan in infer outputs!')
        check = torch.isnan(infer_outputs)
        if True in check:
            print('nan in state of infer outputs!')
         ## concat tile and context
        sequence_output = infer_outputs

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=infer_inputs.hidden_states,
            attentions=infer_inputs.attentions,
        )
