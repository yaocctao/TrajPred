import torch


def fusionGate(x,y):
        '''
        :param x: [batch, station, dim]
        :param y: [batch, station, dim]
        :return: [batch, station, dim]
        '''
        y = torch.multiply(torch.sigmoid(y) , y)
        x = torch.multiply(torch.sigmoid(x) , x)
        h = torch.add(0.5 * x, 0.5 * y)
        # z = torch.sigmoid(torch.mul(x, y))
        # h = torch.add(torch.mul(z, x), torch.mul(1 - z, y))
        return h