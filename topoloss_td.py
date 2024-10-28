import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalLoss(nn.Module):
    def __init__(self):
        super(TopologicalLoss, self).__init__()
        self.kernel01 = torch.tensor([[1, 1, 1],
                                      [0, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel02 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel03 = torch.tensor([[1, 1, 1],
                                      [1, 0, 0],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel04 = torch.tensor([[1, 0, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel05 = torch.tensor([[0, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel06 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel07 = torch.tensor([[1, 1, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel08 = torch.tensor([[1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        
        self.kernel1 = torch.Tensor(  [[0, 0, 1, 1], 
                                       [0, 0, 0, 1], 
                                       [1, 0, 0, 1],
                                       [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel2 = torch.Tensor(  [[1, 1, 0, 0], 
                                      [1, 0, 0, 0], 
                                      [1, 0, 0, 1],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel3 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 1], 
                                      [0, 0, 0, 1],
                                      [0, 0, 1, 1]]).view(1,1,4,4)

        self.kernel4 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 1], 
                                      [1, 0, 0, 0],
                                      [1, 1, 0, 0]]).view(1,1,4,4)

        self.kernel5 = torch.Tensor(  [[1, 0, 0, 1], 
                                      [1, 0, 0, 1], 
                                      [1, 0, 0, 1],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel6 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [0, 0, 0, 1], 
                                      [0, 0, 0, 1],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        self.kernel7 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 1], 
                                      [1, 0, 0, 1],
                                      [1, 0, 0, 1]]).view(1,1,4,4)

        self.kernel8 = torch.Tensor(  [[1, 1, 1, 1], 
                                      [1, 0, 0, 0], 
                                      [1, 0, 0, 0],
                                      [1, 1, 1, 1]]).view(1,1,4,4)

        
        self.kernel10 = torch.tensor([[0, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel20 = torch.tensor([[1, 1, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel30 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 1, 0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel40 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel50 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 0, 0, 0, 0],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel60 = torch.tensor([[1, 1, 1, 1, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [0, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel70 = torch.tensor([[1, 1, 1, 1, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel80 = torch.tensor([[1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 1],
                                      [1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.kernel100 = torch.tensor([[0, 0, 0, 0, 1, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel200 = torch.tensor([[1, 1, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel300 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 1, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel400 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel500 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel600 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel700 = torch.tensor([[1, 1, 1, 1, 1, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.kernel800 = torch.tensor([[1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 0, 0, 0, 0, 1],
                                       [1, 1, 1, 1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        
        self.kernels = [
            self.kernel01, self.kernel02, self.kernel03, self.kernel04, self.kernel05, self.kernel06, self.kernel07, self.kernel08, 
            self.kernel1, self.kernel2, self.kernel3, self.kernel4, self.kernel5, self.kernel6, self.kernel7, self.kernel8, 
            self.kernel10, self.kernel20, self.kernel30, self.kernel40, self.kernel50, self.kernel60, self.kernel70, self.kernel80, 
            self.kernel100, self.kernel200, self.kernel300, self.kernel400, self.kernel500, self.kernel600, self.kernel700, self.kernel800]

    def soft_threshold(self, x, threshold, alpha=10.0):
        x_clipped = torch.clamp(-alpha * (x - threshold), -10, 10)
        return torch.sigmoid(x_clipped)
    
    def custom_sigmoid2(self, mask, beta=0.5, alpha=10):
        return 1 / (1 + torch.exp(-alpha * (mask - beta)))

    def endpoints(self, mask, kernel, padded=True):
        kernel_size = kernel.shape[-1]
        pad = kernel_size // 2
        neighbors_count = F.conv2d(mask, kernel, padding=pad)
        if neighbors_count.shape != mask.shape:
            neighbors_count = neighbors_count[:, :, :mask.shape[2], :mask.shape[3]]
        line_ends = self.soft_threshold(neighbors_count, 1.0) * mask
        if padded:
            line_ends = line_ends[:, :, 1:-1, 1:-1]
        return line_ends

    def critical_points_mapper(self, t, mode):
        epss = torch.zeros_like(t)
        if mode == 'thick':
            ker_list = set(range(len(self.kernels)))
        elif mode == 'thin':
            ker_list = set(range(8))
        padding = (1, 1, 1, 1)
        padded_t = F.pad(t, padding, 'constant', 1)
        if t.is_cuda:
            self.kernels = [k.cuda() for k in self.kernels]
        for i, kernel in enumerate(self.kernels, start=1):
            if i in ker_list:
                eps = self.endpoints(padded_t, kernel=kernel, padded=True)
                epss += eps
        mapped = self.custom_sigmoid2(epss)
        return mapped


    def dialate_tensor(self, input_tensor):
        dilation_size = 3
        dia_kernel = torch.ones(1, 1, dilation_size, dilation_size).cuda()
        dilated_tensor = F.conv2d(input_tensor, dia_kernel, padding=dilation_size // 2)
        dilated_tensor = torch.clamp(dilated_tensor, 0, 1)
        return dilated_tensor

    def erode(self, input_tensor, beta=0.7, alpha=10):
        ker = torch.tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        t = F.conv2d(input_tensor, ker, padding=1)
        eroded = self.custom_sigmoid2(t / (t.max() + 1e-8), beta=beta, alpha=alpha)
        return eroded

    def get_cp(self, tensor, mode):
        t_cp0 = self.critical_points_mapper(tensor, mode=mode)
        t_cp = self.dialate_tensor(t_cp0)
        return t_cp

    def forward(self, pred, gt, num_erosions, mode):
        gt_eroded = [gt]
        p_eroded = [pred]
        for _ in range(num_erosions):
            gt_eroded.append(self.erode(gt_eroded[-1]))
            p_eroded.append(self.erode(p_eroded[-1]))
        diffs = []
        for i in range(num_erosions + 1):
            p_cp = self.get_cp(p_eroded[i], mode=mode)
            gt_cp = self.get_cp(gt_eroded[i], mode=mode)
            diffs.append((p_cp - gt_cp) ** 2)
            
            p_cp_k = self.get_cp(p_eroded[i].max() - p_eroded[i], mode=mode)
            gt_cp_k = self.get_cp(gt_eroded[i].max() - gt_eroded[i], mode=mode)
            diffs.append((p_cp_k - gt_cp_k) ** 2)
        loss_topo = sum(diff.sum() for diff in diffs) / pred.shape[0]
        return [loss_topo, [gt, pred]]

class CustSigmoid(nn.Module):
    def __init__(self, alpha=25, beta=0.5):
        super(CustSigmoid, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * (x - self.beta)))


