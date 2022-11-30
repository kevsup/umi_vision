from sleep_classifier import *

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Lambda(cropBottomLeft), transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(root, transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle=True, num_workers=0)

    class_names = ('sleep', 'wake')
    model = torch.load('sleep_model.pth')
    visualize_model(model, dataloader, class_names)