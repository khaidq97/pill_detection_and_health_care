import torch.nn as nn
import torch
import transformers

class DiagnoseEncoder(nn.Module):
    def __init__(self, BERT_MODEL):
        super(DiagnoseEncoder, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(
                BERT_MODEL, output_hidden_states=True
            )
        self.lstm_units = 768
        self.num_recurrent_layers = 1
        self.bidirectional = False

        self.lstm = nn.LSTM(input_size=768,
                                hidden_size=self.lstm_units,
                                num_layers=self.num_recurrent_layers,
                                bidirectional=self.bidirectional,
                                batch_first=True)
            
        self.dropout = nn.Dropout(0.2)  
        self.out = nn.Linear(768*2, 256)

    def forward(self, diagnose_input):
        input_ids, attention_mask, token_type_ids = diagnose_input[:,0], diagnose_input[:,1], diagnose_input[:,2]
        sequence_output, _ , hidden_outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict = False
                )
    
        if self.bidirectional:
            n = 2
        else: n = 1

        h0 = torch.zeros(self.num_recurrent_layers * n,       # (L * 2 OR L, B, H)
                        input_ids.shape[0],
                        self.lstm_units).to(sequence_output.device)
        c0 = torch.zeros(self.num_recurrent_layers * n,        # (L * 2 OR L, B, H)
                        input_ids.shape[0],
                        self.lstm_units).to(sequence_output.device)

        sequence_output, _ = self.lstm(sequence_output, (h0, c0))

        avg_pool = torch.mean(sequence_output, 1)
        max_pool, _ = torch.max(sequence_output, 1)        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        output = self.dropout(h_conc)
        logits = self.out(output)
        
        return logits