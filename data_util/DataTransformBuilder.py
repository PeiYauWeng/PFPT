import torchvision.transforms as transforms

def build_data_transform(normalize_stats, resize, centercrop_size,
                         interpolation_type=transforms.InterpolationMode.BILINEAR):
    transform = transforms.Compose([transforms.Resize(resize, interpolation=interpolation_type),
                                    transforms.CenterCrop(size=centercrop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*normalize_stats,inplace=True)])
    return transform