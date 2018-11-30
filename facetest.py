import sys
import dlib
import skimage.draw
import skimage.io
import cv2 as cv

path = 'shape_predictor_68_face_landmarks.dat'
load_name = sys.argv[1]
save_name = sys.argv[2]

detect = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)

img = skimage.io.imread(load_name)
nums = detect(img, 1)
print('Number of faces detected: {}'.format(len(nums)))
for i, d in enumerate(nums):
    r0, c0, r1, c1 = d.top(), d.left(), d.bottom(), d.right()
    skimage.draw.set_color(img, skimage.draw.line(r0, c0, r0, c1), (255, 0, 0))
    skimage.draw.set_color(img, skimage.draw.line(r0, c1, r1, c1), (255, 0, 0))
    skimage.draw.set_color(img, skimage.draw.line(r1, c1, r1, c0), (255, 0, 0))
    skimage.draw.set_color(img, skimage.draw.line(r1, c0, r0, c0), (255, 0, 0))

    mark = [(p.x, p.y) for p in landmark(img, d).parts()]
    print('Part 1: {}, Part 2: {} ...'.format(mark[0], mark[1]))
    for i, pos in enumerate(mark):
        print(i)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, str(i+1), pos, font, 0.8, (0, 0, 255), 1,cv.LINE_AA)
        #skimage.draw.set_color(img, skimage.draw.circle(pos[1], pos[0], 2), (0, 0, 255))


skimage.io.imsave(save_name, img)


        

