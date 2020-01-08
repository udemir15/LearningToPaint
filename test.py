import argparse
import os
from DRL.actor import *
from Renderer.network import *
from Renderer.stroke_gen import *


def decode(x, canvas):
    x = x.view(-1, 13)
    stroke_params = 1 - renderer(x[:, :10])
    stroke_params = stroke_params.view(-1, width, width, 1)
    color_params = stroke_params * x[:, -3:].view(-1, 1, 1, 3)
    stroke_params = stroke_params.permute(0, 3, 1, 2)
    color_params = color_params.permute(0, 3, 1, 2)
    stroke_params = stroke_params.view(-1, 5, 1, width, width)
    color_params = color_params.view(-1, 5, 3, width, width)
    results = []
    for iter in np.arange(5):
        canvas = canvas * (1 - stroke_params[:, iter]) + color_params[:, iter]
        results.append(canvas)
    return canvas, results


def small2large(x):
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x


def large2small(x):
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(num_canvas, width, width, 3)
    return x


def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0:
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[
            tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img


def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy()  # d * d, 3, width, width
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, image_shape_origin)
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)


if __name__ == '__main__':

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = 128

    parser = argparse.ArgumentParser(description='Learning to Paint')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--actor', default='./model/Paint-run1/actor.pkl', type=str, help='Actor model')
    parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
    parser.add_argument('--img', default='image/test.png', type=str, help='test image')
    parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
    parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
    args = parser.parse_args()

    num_canvas = args.divide * args.divide
    remaining_strokes = torch.ones([1, 1, width, width], dtype=torch.float32).to(dev)
    image = cv2.imread(args.img, cv2.IMREAD_COLOR)
    image_shape_origin = (image.shape[1], image.shape[0])

    coordinates = torch.zeros([1, 2, width, width])
    for iter in np.arange(width):
        for j in np.arange(width):
            coordinates[0, 0, iter, j] = iter / (width - 1.)
            coordinates[0, 1, iter, j] = j / (width - 1.)
    coordinates = coordinates.to(dev)

    renderer = NeuralRenderer()
    renderer.load_state_dict(torch.load(args.renderer))

    actor = ResNet(9, 18, 65)
    actor.load_state_dict(torch.load(args.actor))
    actor = actor.to(dev).eval()
    renderer = renderer.to(dev).eval()

    current_canvas = torch.zeros([1, 3, width, width]).to(dev)

    patch_img = cv2.resize(image, (width * args.divide, width * args.divide))
    patch_img = large2small(patch_img)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(dev).float() / 255.

    image = cv2.resize(image, (width, width))
    image = image.reshape(1, width, width, 3)
    image = np.transpose(image, (0, 3, 1, 2))
    image = torch.tensor(image).to(dev).float() / 255.

    if not os.path.isdir('output'):
        os.system('mkdir output')

    with torch.no_grad():
        if args.divide != 1:
            args.max_step = args.max_step // 2
        for iter in np.arange(args.max_step):
            step_num = remaining_strokes * iter / args.max_step
            stroke_params = actor(torch.cat([current_canvas, image, step_num, coordinates], 1))
            current_canvas, result = decode(stroke_params, current_canvas)
            print('canvas step {}, L2Loss = {}'.format(iter, ((current_canvas - image) ** 2).mean()))
            for j in np.arange(5):
                save_img(result[j], args.imgid)
                args.imgid += 1
        if args.divide != 1:
            current_canvas = current_canvas[0].detach().cpu().numpy()
            current_canvas = np.transpose(current_canvas, (1, 2, 0))
            current_canvas = cv2.resize(current_canvas, (width * args.divide, width * args.divide))
            current_canvas = large2small(current_canvas)
            current_canvas = np.transpose(current_canvas, (0, 3, 1, 2))
            current_canvas = torch.tensor(current_canvas).to(dev).float()
            coordinates = coordinates.expand(num_canvas, 2, width, width)
            remaining_strokes = remaining_strokes.expand(num_canvas, 1, width, width)
            for iter in range(args.max_step):
                step_num = remaining_strokes * iter / args.max_step
                stroke_params = actor(torch.cat([current_canvas, patch_img, step_num, coordinates], 1))
                current_canvas, result = decode(stroke_params, current_canvas)
                print('divided canvas step {}, L2Loss = {}'.format(iter, ((current_canvas - patch_img) ** 2).mean()))
                for j in range(5):
                    save_img(result[j], args.imgid, True)
                    args.imgid += 1
