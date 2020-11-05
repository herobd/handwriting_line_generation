#from MetaInit: Initialzing learning by learning to initialize
import torch
def gradient_quotient(loss, params, eps=1e-5):
    grad = torch.autograd.grad(loss,
        params, retain_graph=True, create_graph=True)
    prod = torch.autograd.grad(sum([(g**2).sum() / 2 for g in grad]),
        params, retain_graph=True, create_graph=True)
    out = sum([((g - p) / (g + eps * (2*(g >= 0).float() - 1).detach())
        - 1).abs().sum() for g, p in zip(grad, prod)])
    return out / sum([p.data.nelement() for p in params])

def metainit(model, criterion, x_size, y_size, lr=0.1,
    momentum=0.9, steps=500, eps=1e-5):
    model.eval()
    params = [p for p in model.parameters()
        if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)
    for i in range(steps):
        input = torch.Tensor(*x_size).normal_(0, 1).cuda()
        target = torch.randint(0, y_size, (x_size[0],)).cuda()
        loss = criterion(model(input), target)
        gq = gradient_quotient(loss, list(model.parameters()), eps)
        grad = torch.autograd.grad(gq, params)
        for j, (p, g_all) in enumerate(zip(params, grad)):
            norm = p.data.norm().item()
            g = torch.sign((p.data * g_all).sum() / norm)
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            p.data.mul_(new_norm / norm)
        print("%d/GQ = %.2f" % (i, gq.item()))

def metainitRecog(model, criterion, x_size, y_size, lr=0.1,
    momentum=0.9, steps=500, eps=1e-5):
    batch_size = x_size[0]
    #model.eval()
    params = [p for p in model.parameters()
        if p.requires_grad and len(p.size()) >= 2]
    memory = [0] * len(params)
    for i in range(steps):
        input = torch.Tensor(*x_size).normal_(0, 1).cuda()
        target = torch.randint(0, y_size, (batch_size,x_size[3]//(x_size[2]//3)+3)).cuda()
        output = model(input)
        pred_size = torch.IntTensor([output.size(0)] * batch_size)
        label_lengths = torch.randint(x_size[3]//(x_size[2]//3)-3, x_size[3]//(x_size[2]//3)+3, (batch_size,))
        loss = criterion(output, target,pred_size,label_lengths)
        gq = gradient_quotient(loss, list(model.parameters()), eps)
        grad = torch.autograd.grad(gq, params)
        for j, (p, g_all) in enumerate(zip(params, grad)):
            norm = p.data.norm().item()
            g = torch.sign((p.data * g_all).sum() / norm)
            memory[j] = momentum * memory[j] - lr * g.item()
            new_norm = norm + memory[j]
            p.data.mul_(new_norm / norm)
        print("%d/GQ = %.2f" % (i, gq.item()))
