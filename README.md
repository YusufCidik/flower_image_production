# Botanical Diffusion: Tesla T4 ile 128x128 Çiçek Üretimi

Bu proje, NVIDIA Tesla T4 GPU üzerinde kısıtlı kaynaklarla yüksek çözünürlüklü (128x128) ve profesyonel kalitede botanik görseller üretmek için geliştirildi. Flowers102 veri seti üzerinde eğitilen model, bellek optimizasyon teknikleri sayesinde 16GB VRAM'in altında oldukça verimli bir performans sergilemektedir. 🌸

## Öne Çıkanlar

*   **Yüksek Çözünürlük:** Standart düşük çözünürlüklü modeller yerine, 128x128 native çözünürlük ile daha detaylı dokular.
*   **Bellek Dostu Mimari:** 8-bit optimizer kullanarak yaklaşık 10GB VRAM kullanımı ile Tesla T4 üzerinde stabil çalışma.
*   **Hızlı Eğitim:** Eğitim aşamasında saniyede ~1.25 iterasyon (it/s) hızına ulaşan optimize edilmiş döngü.
*   **Düşük FID Skorları:** Kısa sürede 8.11 gibi başarılı FID değerlerine ulaşan hızlı öğrenme kapasitesi.

## Donanım ve Performans Bilgileri

Modelin eğitimi sırasında kullanılan sistem verileri ve kaynak tüketimi şu şekildedir:

*   **GPU:** NVIDIA Tesla T4
*   **CUDA Sürümü:** 13.0
*   **Sürücü Versiyonu:** 580.82.07
*   **VRAM Kullanımı:** ~9950 MiB (Belleğin yaklaşık %65'i aktif kullanıldı)
*   **Eğitim Hızı:** Her epoch ortalama 50 saniye sürmektedir.

## Kurulum

Projeyi kendi ortamınızda çalıştırmak için gerekli kütüphaneleri yükleyin:

```bash
pip install torch torchvision diffusers accelerate bitsandbytes torchmetrics tqdm pillow matplotlib
```

## Kullanım

Model dosyaları 25MB sınırını aştığı için harici bir link üzerinden sunulmaktadır.

**1. Modeli İndirin**
Aşağıdaki linkten model klasörünü indirip projenizin ana dizinine `t4_photo_pro_model` ismiyle kaydedin:

🔗 **[https://drive.google.com/drive/folders/1CxsTPksKvDzpzd6weBpO1ywbWi_xQW9J?usp=sharing]**

**2. Görsel Üretimini Başlatın**
Eğitilmiş ağırlıkları kullanarak görsel üretmek için şu kodu kullanabilirsiniz:

```python
from diffusers import DDPMPipeline, DDIMScheduler
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modeli yükle
model_yolu = "./t4_photo_pro_model"
pipe = DDPMPipeline.from_pretrained(model_yolu, torch_dtype=torch.float16).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Üretim (Inference)
with torch.no_grad():
    image = pipe(num_inference_steps=100).images[0]
    image.save("botanik_cicek.png")
    image.show()
```

## Eğitim Süreci ve Metrikler

Eğitim sırasında her epoch sonunda hesaplanan FID (Frechet Inception Distance) skorları, modelin görsel kalitesindeki artışı belgelemektedir. FID skoru düştükçe modelin ürettiği görseller gerçeğe daha fazla yaklaşmaktadır.

| Epoch | Loss (Kayıp) | FID Skoru |
| :--- | :--- | :--- |
| Epoch 0 | 0.0915 | 28.2811 |
| Epoch 1 | 0.0821 | 15.0218 |
| Epoch 2 | 0.0470 | 14.1721 |
| **Epoch 3** | **0.1230** | **8.1103** |

*Eğitim boyunca ortalama kayıp (loss) değerleri 0.04 ile 0.10 arasında stabilize olmuştur.*

## Dosya Yapısı

*   `train.py`: Tesla T4 optimizasyonlu eğitim scripti.
*   `inference.py`: Görsel üretimi ve iyileştirme arayüzü.
*   `t4_photo_pro_model/`: Eğitilmiş model dosyaları (Config, UNet ağırlıkları).

---
**Geliştirici:** [Yusuf Cidik]
Projeyle ilgili sorularınız için bir Issue açabilir veya iletişime geçebilirsiniz.
