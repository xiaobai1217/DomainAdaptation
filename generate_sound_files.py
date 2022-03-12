import moviepy.editor as mp
import librosa
import soundfile
import glob
import subprocess
import os
import multiprocessing

def obtain_list(source_path):
    files = []
    txt = glob.glob(source_path + '/*.MP4') # '/*.flac'
    for item in txt:
        files.append(item)
    return files

def convert(v, output_path):
    subprocess.check_call([
    'ffmpeg',
    '-n',
    '-i', v,
    '-acodec', 'pcm_s16le',
    '-ac','1',
    '-ar','16000',
    output_path + '%s.wav' % v.split('/')[-1][:-4]])

split_list = ['train', 'test']
domain_list = ['D1', 'D2', 'D3']
for domain in domain_list:
    for split in split_list:
        source_path = '/home/xxx/data/EPIC_KITCHENS_UDA/Videos/'+split+'/'+domain
        output_path = '/home/xxx/data/EPIC_KITCHENS_UDA/AudioVGGSound/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path = output_path + split
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_path = output_path + '/'+domain +'/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        file_list = obtain_list(source_path)
        for i, file1 in enumerate(file_list):
            convert(file1, output_path)




# def convert(v):
#     subprocess.check_call([
#     'ffmpeg',
#     '-n',
#     '-i', v,
#     '-acodec', 'pcm_s16le',
#     '-ac','1',
#     '-ar','48000',
#     '-format', 's16le',
#     output_path + '%s.wav' % v.split('/')[-1][:-4]])




# p = multiprocessing.Pool(16)
# p.map(convert, obtain_list())
