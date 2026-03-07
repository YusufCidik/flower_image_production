# Botanical Diffusion Pro: Tesla T4 Optimize Edilmiş Çiçek Üretimi

Bu proje, kısıtlı donanım kaynaklarıyla (NVIDIA Tesla T4) yüksek kalitede botanik görseller üretmek amacıyla geliştirilmiş bir Diffusion modelidir. Standart modellerin aksine 128x128 native çözünürlükte eğitilmiş ve Gradio tabanlı modern bir kullanıcı arayüzüyle desteklenmiştir. 

Model, eğitim sırasında `bitsandbytes` 8-bit optimizer kullanarak bellek verimliliğini maksimize etmiş, üretim aşamasında ise DDIM zamanlayıcı ve post-processing (keskinleştirme/kontrast) teknikleriyle görsel kaliteyi 512x512 seviyesine taşımıştır.

## Öne Çıkan Özellikler

*   **Gelişmiş UNet Mimarisi:** 128x128 çözünürlükte detay kaybını önleyen derin katman yapısı.
*   **Hızlı ve Verimli:** DDIM Scheduler sayesinde 100-120 adımda keskin ve kaliteli sonuçlar.
*   **Kullanıcı Dostu Arayüz:** Gradio üzerinden canlı parametre kontrolü (Seed, Steps, Keskinlik, Kontrast).
*   **Post-Processing:** Üretilen görselleri otomatik olarak 512x512 boyutuna ölçekleyen ve fotoğraf kalitesini artıran filtreler.

## Donanım ve Performans Verileri

Eğitim sürecinde Tesla T4 GPU üzerinde elde edilen veriler şu şekildedir:
*   **VRAM Kullanımı:** ~9.9 GB (8-bit optimizasyon ile).
*   **Eğitim Hızı:** ~1.25 it/s.
*   **En İyi FID Skoru:** 4.07 (31. Epoch).

## Kurulum

Projeyi çalıştırmak için gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install --upgrade torch torchvision diffusers accelerate safetensors matplotlib pillow bitsandbytes gradio
```

## Model Dosyalarını Edinme

Model ağırlıkları ve yapılandırma dosyası GitHub limitleri nedeniyle dış bağlantıda tutulmaktadır. 

1. Aşağıdaki linkten klasörü indirin:
   🔗**https://drive.google.com/drive/folders/1YDv4ZsRLtXzEUwKqpM0cVl9vSkw1_Yc2**
2. İndirdiğiniz `final_botanic_model.safetensors` ve `botanical_final.json` dosyalarını projenizin dizinine yerleştirin.
3. Kod içerisindeki `yapilandirma_yolu` ve `model_agirlik_yolu` değişkenlerini kendi dosya yolunuza göre güncelleyin.

## Kullanım ve Arayüz

Modeli başlatmak için hazırlanan scripti çalıştırdığınızda yerel ağınızda bir web arayüzü oluşacaktır.

```python
python app.py
```

### Arayüz Parametreleri
*   **Seed:** Görselin temel formunu belirler. Aynı sayı ile aynı görseli tekrar üretebilirsiniz.
*   **Üretim Adımı (Steps):** Modelin görseli oluşturma süresi. 100-120 adım arası en dengeli sonuçları verir.
*   **Keskinlik Seviyesi:** Upscale sonrası oluşan yumuşaklığı giderir, detayları belirginleştirir.
*   **Kontrast Oranı:** Görselin renk doygunluğunu ve derinliğini profesyonel fotoğraf seviyesine çeker.

## Eğitim Günlüğü

Eğitim sürecindeki FID (Frechet Inception Distance) skorları modelin gelişimini göstermektedir:

| Epoch | FID Skoru | Durum |
| :--- | :--- | :--- |
| 0 | 28.28 | Başlangıç |
| 15 | 6.24 | Formlar belirginleşti |
| **31** | **4.07** | **En yüksek kalite** |
| 73 | 5.52 | Doygunluk artışı |

## Dosya Yapısı

*   `app.py`: Gradio arayüzünü başlatan ana dosya.
*   `botanical_final.json`: UNet model mimarisi yapılandırması.
*   `final_botanic_model.safetensors`: Eğitilmiş ağırlık dosyası.

---
**Geliştirici:** [Yusuf Cidik]
Sorularınız veya katkılarınız için Issue üzerinden iletişime geçebilirsiniz.

**Çıktılar**
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/b9f82e91-033a-44d6-8c69-5c82694fd7cd" />
<img width="674" height="639" alt="image" src="https://github.com/user-attachments/assets/f7b434c2-9c0b-485c-bbbf-bb93fe3c39cb" />
Epoch 0: 100%
 63/63 [00:52<00:00,  1.24it/s, loss=0.0915]
100%
 50/50 [00:03<00:00, 13.26it/s]
✨ Epoch 0 Bitti | Güncel FID: 28.2811

Epoch 1: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0821]
100%
 50/50 [00:03<00:00, 11.64it/s]
✨ Epoch 1 Bitti | Güncel FID: 15.0218

Epoch 2: 100%
 63/63 [00:51<00:00,  1.23it/s, loss=0.047]
100%
 50/50 [00:03<00:00, 13.14it/s]
✨ Epoch 2 Bitti | Güncel FID: 14.1721

Epoch 3: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.123]
100%
 50/50 [00:03<00:00, 12.48it/s]
✨ Epoch 3 Bitti | Güncel FID: 8.1103

Epoch 4: 100%
 63/63 [00:51<00:00,  1.23it/s, loss=0.083]
100%
 50/50 [00:03<00:00, 12.93it/s]
✨ Epoch 4 Bitti | Güncel FID: 16.6659

Epoch 5: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0844]
100%
 50/50 [00:03<00:00, 12.71it/s]
✨ Epoch 5 Bitti | Güncel FID: 15.3149

Epoch 6: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0931]
100%
 50/50 [00:03<00:00, 13.13it/s]
✨ Epoch 6 Bitti | Güncel FID: 10.2074

Epoch 7: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0934]
100%
 50/50 [00:03<00:00, 13.18it/s]
✨ Epoch 7 Bitti | Güncel FID: 21.0769

Epoch 8: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0591]
100%
 50/50 [00:03<00:00, 13.10it/s]
✨ Epoch 8 Bitti | Güncel FID: 5.9902

Epoch 9: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0366]
100%
 50/50 [00:03<00:00, 13.17it/s]
✨ Epoch 9 Bitti | Güncel FID: 4.8320

Epoch 10: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0767]
100%
 50/50 [00:03<00:00, 13.12it/s]
✨ Epoch 10 Bitti | Güncel FID: 19.6913

Epoch 11: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0668]
100%
 50/50 [00:03<00:00, 12.87it/s]
✨ Epoch 11 Bitti | Güncel FID: 8.4837

Epoch 12: 100%
 63/63 [00:51<00:00,  1.25it/s, loss=0.045]
100%
 50/50 [00:03<00:00, 13.15it/s]
✨ Epoch 12 Bitti | Güncel FID: 25.9451

Epoch 13: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0497]
100%
 50/50 [00:03<00:00, 13.15it/s]
✨ Epoch 13 Bitti | Güncel FID: 8.1535

Epoch 14: 100%
 63/63 [00:51<00:00,  1.25it/s, loss=0.0476]
100%
 50/50 [00:03<00:00, 12.80it/s]
✨ Epoch 14 Bitti | Güncel FID: 10.2754

Epoch 15: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0738]
100%
 50/50 [00:03<00:00, 12.91it/s]
✨ Epoch 15 Bitti | Güncel FID: 6.2491

Epoch 16: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0374]
100%
 50/50 [00:04<00:00, 13.10it/s]
✨ Epoch 16 Bitti | Güncel FID: 21.4591

Epoch 17: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0528]
100%
 50/50 [00:03<00:00, 13.15it/s]
✨ Epoch 17 Bitti | Güncel FID: 6.4339

Epoch 18: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0614]
100%
 50/50 [00:04<00:00, 13.08it/s]
✨ Epoch 18 Bitti | Güncel FID: 7.8042

Epoch 19: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0457]
100%
 50/50 [00:03<00:00, 13.13it/s]
✨ Epoch 19 Bitti | Güncel FID: 7.0154

Epoch 20: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.051]
100%
 50/50 [00:04<00:00, 13.05it/s]
✨ Epoch 20 Bitti | Güncel FID: 7.9500

Epoch 21: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.042]
100%
 50/50 [00:03<00:00, 13.18it/s]
✨ Epoch 21 Bitti | Güncel FID: 26.7750

Epoch 22: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0268]
100%
 50/50 [00:04<00:00, 13.01it/s]
✨ Epoch 22 Bitti | Güncel FID: 4.2950

Epoch 23: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0254]
100%
 50/50 [00:03<00:00, 13.02it/s]
✨ Epoch 23 Bitti | Güncel FID: 11.8616

Epoch 24: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0652]
100%
 50/50 [00:04<00:00, 13.01it/s]
✨ Epoch 24 Bitti | Güncel FID: 8.6093

Epoch 25: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.032]
100%
 50/50 [00:03<00:00, 13.06it/s]
✨ Epoch 25 Bitti | Güncel FID: 10.9438

Epoch 26: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0394]
100%
 50/50 [00:04<00:00, 12.80it/s]
✨ Epoch 26 Bitti | Güncel FID: 4.1891

Epoch 27: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0254]
100%
 50/50 [00:03<00:00, 13.15it/s]
✨ Epoch 27 Bitti | Güncel FID: 7.1871

Epoch 28: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0653]
100%
 50/50 [00:04<00:00, 12.85it/s]
✨ Epoch 28 Bitti | Güncel FID: 7.6703

Epoch 29: 100%
 63/63 [00:51<00:00,  1.25it/s, loss=0.0277]
100%
 50/50 [00:03<00:00, 13.01it/s]
✨ Epoch 29 Bitti | Güncel FID: 12.1320

Epoch 30: 100%
 63/63 [00:51<00:00,  1.27it/s, loss=0.0603]
100%
 50/50 [00:04<00:00, 12.48it/s]
✨ Epoch 30 Bitti | Güncel FID: 7.0679

Epoch 31: 100%
 63/63 [00:51<00:00,  1.25it/s, loss=0.026]
100%
 50/50 [00:03<00:00, 13.08it/s]
✨ Epoch 31 Bitti | Güncel FID: 4.0776

Epoch 32: 100%
 63/63 [00:51<00:00,  1.26it/s, loss=0.0502]
100%
 50/50 [00:04<00:00, 12.90it/s]
✨ Epoch 32 Bitti | Güncel FID: 18.1830

Epoch 33: 100%
 63/63 [00:51<00:00,  1.25it/s, loss=0.043]
100%
 50/50 [00:03<00:00, 13.09it/s]
✨ Epoch 33 Bitti | Güncel FID: 9.7198

Epoch 34: 100%
...(etc.)
