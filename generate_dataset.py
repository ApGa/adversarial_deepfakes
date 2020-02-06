"""
File: generate_dataset.py -- Deepfake Generation Script
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

from models import FaceTranslationGANInferenceModel
from face_toolbox_keras.models.verifier.face_verifier import FaceVerifier
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
from face_toolbox_keras.models.detector.iris_detector import IrisDetector
import numpy as np
from utils import utils
from matplotlib import pyplot as plt
from itertools import permutations
from tqdm import tqdm
import glob
import random

TRUNCATE = 10000
'''
fv = FaceVerifier(classes=512)
fp = face_parser.FaceParser()
fd = face_detector.FaceAlignmentDetector()
idet = IrisDetector()

model = FaceTranslationGANInferenceModel()
'''
image_locations = glob.glob("C:\\Users\\Apurva\\Desktop\\C_Projects\\img_align_celeba\\img_align_celeba\\*")[18832:TRUNCATE + 10000]

#image_locations = ["C:\\Users\\Apurva\\Desktop\\C_Projects\\img_align_celeba\\img_align_celeba\\004524.jpg", "C:\\Users\\Apurva\\Desktop\\C_Projects\\img_align_celeba\\img_align_celeba\\009501.jpg"]


#fn_src = "source.jpg"
#fns_tar = ["target.jpg"]
'''
for fn_src, fn_tar in tqdm(permutations(image_locations, 2)):
	try:
		src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(fn_src, fd, fp, idet)
		tar, emb_tar = utils.get_tar_inputs(fn_tar, fd, fv)

		out = model.inference(src, mask, tar, emb_tar)

		result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
		#plt.imshow(result_face)
		#plt.show()
		result_img = utils.post_process_result(fn_src, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
		plt.imshow(result_img)
		#plt.show()
		fn_src_name = (fn_src.split("\\")[-1]).split(".")[0]
		fn_tar_name = (fn_tar.split("\\")[-1]).split(".")[0]
		plt.savefig('images/celeba_swapped/{}_{}.jpg'.format(fn_src_name, fn_tar_name))
	except: 
		print(fn_src, fn_tar)
'''
'''
for fn_src in image_locations:
	for _ in range(5):
		err = False
		try:
			fn_tar = random.choice(image_locations)
			while(fn_tar == fn_src):
				fn_tar = random.choice(image_locations)
			src, mask, aligned_im, (x0, y0, x1, y1), landmarks = utils.get_src_inputs(fn_src, fd, fp, idet)
			tar, emb_tar = utils.get_tar_inputs(fn_tar, fd, fv)

			out = model.inference(src, mask, tar, emb_tar)

			result_face = np.squeeze(((out[0] + 1) * 255 / 2).astype(np.uint8))
			#plt.imshow(result_face)
			#plt.show()
			#result_img = utils.post_process_result(fn_src, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
			result_img = utils.post_process_result(fn_src, fd, result_face, aligned_im, src, x0, y0, x1, y1, landmarks)
			plt.imshow(result_img)
			#plt.show()
			fn_src_name = (fn_src.split("\\")[-1]).split(".")[0]
			fn_tar_name = (fn_tar.split("\\")[-1]).split(".")[0]
			plt.savefig('C:/Users/Apurva/Desktop/C_Projects/img_align_celeba_swapped/{}_{}.jpg'.format(fn_src_name, fn_tar_name))
		except:
			err = True 
			print(fn_src, fn_tar)

		if(not err):
			break
'''

for fn_src in image_locations:
	plt.imshow(utils.read_and_resize(fn_src))
	fn_src_name = (fn_src.split("\\")[-1]).split(".")[0]
	plt.savefig('C:/Users/Apurva/Desktop/C_Projects/img_align_celeba_real/{}.jpg'.format(fn_src_name))