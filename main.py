import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import networks.utils as utils
from tqdm import tqdm
import torch.nn.functional as F
from networks.nets import model as sutu_model



def contrastive_loss_3(out_1, out_2, out_3):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    out_3 = F.normalize(out_3, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    out = torch.cat([out_1, out_2, out_3], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(3 * bs, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(3 * bs, -1)
    pos_sim_12 = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    pos_sim_13 = torch.exp(torch.sum(out_1 * out_3, dim=-1) / temp)
    pos_sim_23 = torch.exp(torch.sum(out_2 * out_3, dim=-1) / temp)
    pos_sim = torch.cat([pos_sim_12, pos_sim_23, pos_sim_13], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print('Epoch: {}, AUROC is: {}'.format(0, auc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    center = F.normalize(center, dim=-1)
    center = center.to(device)
    max_auc = 0
    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader_1, optimizer, center, device)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        auc, _ = get_score(model, device, train_loader, test_loader)
        if auc>max_auc:
            max_auc =auc
        print('Epoch: {}, AUROC is: {}, max:{}'.format(epoch + 1, auc, max_auc))


def run_epoch(model, train_loader, optimizer, center, device):
    total_loss, total_num = 0.0, 0
    for ((img1, img2, img_3), _) in tqdm(train_loader, desc='Train...'):

        img1, img2, img3 = img1.to(device), img2.to(device), img_3.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_3 = model(img3)

        out_1 = out_1 - center
        out_2 = out_2 - center
        out_3 = out_3 - center

        center_loss = ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean()) + (out_3 ** 2).sum(dim=1).mean()
        loss = contrastive_loss_3(out_1, out_2, out_3) + center_loss #use
        loss.backward()
        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (imgs, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = sutu_model()
    model = model.to(device)

    train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size)
    train_model(model, train_loader, test_loader, train_loader_1, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--epochs', default=100, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()
    main(args)