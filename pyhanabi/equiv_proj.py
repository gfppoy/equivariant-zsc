from itertools import permutations
import torch

class EquivariantProjection(torch.jit.ScriptModule):
    def __init__(self, sad, train_device):
        super().__init__()
        colours = [0, 1, 2, 3, 4]
        self.symmetries = torch.tensor(list(permutations(colours))[1:])
        self.num_symmetries = len(self.symmetries)+1
        if (sad):
            self.input_perms = torch.eye((713), device=train_device)
        else:
            self.input_perms = torch.eye((658), device=train_device)
        self.output_perms = torch.eye((21), device=train_device)

        # input permutations
        for symm in self.symmetries:
            if (sad):
                perm = torch.eye(713)
            else:
                perm = torch.eye(658)
            
            # partner hand
            card_perm = torch.zeros(25)
            for i,idx in enumerate(symm):
                temp = torch.tensor([0,1,2,3,4], dtype=torch.long)
                temp = 5*i+temp
                card_perm[5*idx:5*(idx+1)] = temp

            hand_perm = torch.zeros(125, dtype=torch.long)
            for i in range(5):
                hand_perm[25*i:25*(i+1)] = card_perm + 25*i

            perm[0:125] = perm[hand_perm]

            # fireworks
            fireworks_perm = card_perm + 167
            perm[167:192] = perm[fireworks_perm.long()]

            # discards
            discards_perm = torch.zeros(50, dtype=torch.long)
            for i,idx in enumerate(symm):
                temp = torch.tensor([203,204,205,206,207,208,209,210,211,212], dtype=torch.long)
                temp = 10*i+temp
                discards_perm[10*idx:10*(idx+1)] = temp

            perm[203:253] = perm[discards_perm]

            # last action
            which_colour_perm = torch.tensor([261,261,261,261,261], dtype=torch.long) + symm
            perm[261:266] = perm[which_colour_perm]
            perm[281:306] = perm[card_perm.long() + 281]

            # V0
            for i in range(10):
                perm[308+35*i:333+35*i] = perm[card_perm.long() + 308+35*i]
                directly_revealed_colour_perm = torch.tensor([333,333,333,333,333], dtype=torch.long) + 35*i + symm
                perm[333+35*i:338+35*i] = perm[directly_revealed_colour_perm]

            # greedy action
            if (sad):
                which_colour_perm = torch.tensor([666,666,666,666,666], dtype=torch.long) + symm
                perm[666:671] = perm[which_colour_perm]
                perm[686:711] = perm[card_perm.long() + 686]

            self.input_perms = torch.cat([self.input_perms, perm.to(train_device)], 1)

        # output permutations
        for symm in self.symmetries:
            perm = torch.eye(21)
            temp = torch.tensor([10,10,10,10,10], dtype=torch.long)
            perm[10:15] = perm[temp + symm]
            self.output_perms = torch.cat([self.output_perms, perm.to(train_device)], 0)


    @torch.jit.script_method
    def project(self, 
            input_weight: torch.Tensor, 
            output_weight: torch.Tensor, 
            output_bias: torch.Tensor): 

        symm_input_weight = torch.matmul(input_weight, self.input_perms).reshape(input_weight.size(0), self.num_symmetries, input_weight.size(1))
        symm_input_weight = torch.sum(symm_input_weight, 1)/self.num_symmetries

        symm_output_weight = torch.matmul(self.output_perms, output_weight).reshape(self.num_symmetries, output_weight.size(0), output_weight.size(1))
        symm_output_weight = torch.sum(symm_output_weight, 0)/self.num_symmetries

        symm_output_bias = torch.matmul(self.output_perms, output_bias).reshape(self.num_symmetries, output_bias.size(0))
        symm_output_bias = torch.sum(symm_output_bias, 0)/self.num_symmetries

        return symm_input_weight, symm_output_weight, symm_output_bias
