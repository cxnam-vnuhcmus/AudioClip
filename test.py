import librosa
import numpy as np
import librosa
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler


def extract_llf_features(audio_data, sr, n_fft, win_length, hop_length):
    # Rút trích đặc trưng âm thanh
    # Âm lượng
    rms = librosa.feature.rms(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)

    # Tần số cơ bản
    chroma = librosa.feature.chroma_stft(n_chroma=17,y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)

    # Tần số biên độ
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)

    # Mức độ biến đổi âm lượng và tần số
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    
    #Poly-features
    poly_features = librosa.feature.poly_features(y=audio_data, sr=sr, n_fft=n_fft, win_length=win_length, hop_length=hop_length, center=False)
    
    # Compute zero-crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=win_length, hop_length=hop_length, center=False)
    
    feats = np.vstack((chroma, #12
                spectral_contrast, #7
                spectral_centroid, #1
                spectral_bandwidth, #1
                spectral_flatness, #1
                spectral_rolloff, #1
                poly_features, #2
                rms, #1
                zcr #1
                )) 

    scaler = StandardScaler()
    feats = scaler.fit_transform(feats.T).T
    
    feats = librosa.power_to_db(feats)
    
    return feats

filepath = '/home/cxnam/Documents/MEAD/M003/audios/front_angry_level_1/002.wav'
sample_rate = 16000
n_fft = 800
n_mels = 128
n_mfcc = 40
win_length = 800
hop_length = 635
audio_data, _ = librosa.load(filepath, sr=sample_rate)
#Thu voi MFCC
mfcc = librosa.feature.mfcc(y=audio_data, 
                                                sr=sample_rate, 
                                                n_mfcc=n_mfcc, 
                                                n_fft=n_fft,
                                                win_length=win_length, 
                                                hop_length=hop_length,
                                                center=False)
mfcc_db = librosa.power_to_db(mfcc)
mfcc_seg = np.transpose(mfcc_db)[10:15,:]
delta_mfcc = librosa.feature.delta(mfcc_seg)
delta2_mfcc = librosa.feature.delta(mfcc_seg, order=2)
combined_mfcc_features = np.concatenate([mfcc_seg, delta_mfcc, delta2_mfcc], axis=1)
combined_mfcc_features = resize(combined_mfcc_features, (32, 64), anti_aliasing=True)

#Thu voi LLFs
llfs_db = extract_llf_features(audio_data, sample_rate, n_fft, win_length, hop_length)
llfs_seg = np.transpose(llfs_db)[10:15,:]
delta_llfs = librosa.feature.delta(llfs_seg)
delta2_llfs = librosa.feature.delta(llfs_seg, order=2)
combined_llfs_features = np.concatenate([llfs_seg, delta_llfs, delta2_llfs], axis=1)
combined_llfs_features = resize(combined_llfs_features, (32, 64), anti_aliasing=True)


combined_features = np.concatenate([combined_mfcc_features, combined_llfs_features], axis=0)
combined_features = resize(combined_features, (224, 224), anti_aliasing=False)

#Save
plt.imshow(combined_features, cmap='magma', aspect='equal', origin='lower')
plt.axis('off')
plt.savefig('llfs_3_2.png', bbox_inches='tight', pad_inches=0)
plt.show()


melspec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=80, fmax=8000)
melspec_db = librosa.power_to_db(melspec, ref=np.max)

mfcc_sum = np.average(mfcc_db, axis=1)
plt.figure(figsize=(10, 6))
print(mfcc.shape)
plt.plot(mfcc_sum)
plt.ylabel('Amplitute')
plt.xlabel('MFCC Coefficients')
plt.title('Mel-Frequency Cepstral Coefficients (MFCC)')
plt.savefig('mfcc.png', bbox_inches='tight', pad_inches=0)
plt.show()