import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, DDIMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import bitsandbytes as bnb
from PIL import Image




# 1. accelerator (t4 turbo - fp16)
accelerator = Accelerator(mixed_precision='fp16')
device = accelerator.device




# zorunlu kanit
print("--- t4 gpu kullanim kaniti ---")
!nvidia-smi




# 2. ayarlar
res = 128 # çözünürlük 128 (fotoğraf kalitesi için)
batch_size = 16 # belleği korumak için (128x128 daha çok vram yer)
epochs = 100




transform = transforms.Compose([
    transforms.Resize((res, res)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.Flowers102(root="./data", download=True, transform=transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)




# 3. yeni: 128x128 fotoğraf kalitesi mimarisi
model = UNet2DModel(
    sample_size=res,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"
    ),
    up_block_types=(
        "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D",
        "UpBlock2D", "UpBlock2D", "UpBlock2D"
    ),
)




optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")




# fid metriği (hız için 64 feature)
fid_metric = FrechetInceptionDistance(feature=64).to(device)




model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)




print(f"128x128 pro-quality eğitimi başladi.")




for epoch in range(epochs):
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"epoch {epoch}")
    for step, (images, _) in enumerate(progress_bar):
        noise = torch.randn(images.shape).to(device)
        timesteps = torch.randint(0, 1000, (images.shape[0],), device=device).long()
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)




        with accelerator.autocast():
            noise_pred = model(noisy_images, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)




        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.set_postfix(loss=loss.item())




    # her epoch'ta fid hesaplama
    model.eval()
    with torch.no_grad():
        # gerçek görüntüleri ekle (batch'ten al)
        real_imgs = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        fid_metric.update(real_imgs, real=True)




        # sahte görüntü üret (hiz için ddim 50 step)
        unwrapped_model = accelerator.unwrap_model(model)
        # ddim scheduler oluştur
        ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
        pipeline = DDPMPipeline(unet=unwrapped_model, scheduler=ddim_scheduler)




        # epoch başina 4 resim üretip fid güncelle (hiz için)
        generated = pipeline(batch_size=4, num_inference_steps=50).images
        fake_tensors = torch.stack([transform(img) for img in generated]).to(device)
        fake_imgs = ((fake_tensors + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        fid_metric.update(fake_imgs, real=False)




        current_fid = fid_metric.compute()
        print(f"\nepoch {epoch} bitti | güncel fid: {current_fid:.4f}")
        fid_metric.reset() # her epoch kendi skorunu göstersin




# final kayit (fp16)
model_to_save = accelerator.unwrap_model(model)
model_to_save.to(torch.float16).save_pretrained("t4_photo_pro_model")
print("128x128 profesyonel model kaydedildi!")
