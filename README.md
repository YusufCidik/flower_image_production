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
   🔗 **https://drive.google.com/drive/folders/1YDv4ZsRLtXzEUwKqpM0cVl9vSkw1_Yc2**
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
**Geliştirici:** [Adın Soyadın]
Sorularınız veya katkılarınız için Issue üzerinden iletişime geçebilirsiniz.
