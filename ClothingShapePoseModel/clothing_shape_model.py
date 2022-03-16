import numpy as np
from models import ClothingShapeModel, ClothingShapeModelMLP

'''
THIS FILE IS DEPRECATED. DO NOT USE IT, JUST FOR ADVISING.
'''

'''
Version 0: basic model proposed in original paper
'''
model = ClothingShapeModel(np.zeros(72), 'clothes', 'clothShirts1', '../../Metown-body/base_body', regress_from_vs = False)
model.drape(np.asarray([0,0,0,0,0,0,0,0,0,0]), name='sample0')
model.drape(np.asarray([0,0,0,0,0,0,0,0,0,0])-1, name='sample-1')
model.drape(np.asarray([0,0,0,0,0,0,0,0,0,0])+1, name='sample1')

'''
Version 1: regress the cloth vertices directly from human vertices rather than human shape parameter
'''
# model = ClothingShapeModel(np.zeros(72), 'clothes', 'clothShirts1', '../../Metown-body/base_body', regress_from_vs = True)
#
# from STAR_utils import STAR_UTILS
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10])
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = None, body_vs = vertices.flatten(), name='sample0')
#
# from STAR_utils import STAR_UTILS
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10]) + 1
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = None, body_vs = vertices.flatten(), name='sample1')
#
# from STAR_utils import STAR_UTILS
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10]) -1
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = None, body_vs = vertices.flatten(), name='sample-1')

'''
Version 2: regress the cloth vertices directly from human vertices rather than human shape parameter using MLP
'''
# model = ClothingShapeModelMLP(np.zeros(72), 'clothes', 'clothShirts1', '../../Metown-body/base_body', regress_from_vs = True)
#
# from STAR_utils import STAR_UTILS
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10])
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = None, body_vs = vertices, name = 'sample0')
#
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10]) + 1
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = None, body_vs = vertices, name = 'sample1')
#
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10]) - 1
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = None, body_vs = vertices, name = 'sample-1')

'''
Version 3: regress the cloth vertices from human shape parameter using MLP
'''
# model = ClothingShapeModelMLP(np.zeros(72), 'clothes', 'clothShirts1', '../../Metown-body/base_body', regress_from_vs = False)
#
# from STAR_utils import STAR_UTILS
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10])
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = shape, body_vs = vertices, name = 'sample0')
#
# from STAR_utils import STAR_UTILS
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10]) +1
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = shape, body_vs = vertices, name = 'sample1')
#
# from STAR_utils import STAR_UTILS
# star = STAR_UTILS()
# pose = np.zeros([1, 72])
# shape = np.zeros([1, 10]) -1
# trans = np.zeros([1, 3])
# vertices = star.star_forward(pose, shape, trans, disps = None, to_numpy = True)
# vertices[:, :, 1] -= vertices[:, :, 1].min()
# model.drape(shape = shape, body_vs = vertices, name = 'sample-1')
