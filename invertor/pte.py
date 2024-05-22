import copy
from typing import Callable, Union, List, Literal, Dict
import cv2
from sklearn.decomposition import PCA
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
import dill
import facenet_pytorch
from face.facealign import LANDMARKS_MEDIAPIPE
from face.faceclass.face import Face
from image.edit import wrinkle_remover
from invertor.utils import read_image
from invertor.loss import LpipsLoss
from invertor.experimental import invert_StyleGAN_experiments


def pivotal_tuning_direction(
    G: torch.nn.Module, # pylint: disable=C0103
    w_pivot: List[torch.Tensor],
    target: List[Union[str, Image.Image, np.array, torch.Tensor, None]],
    wrinkle_targets: List[Union[str, Image.Image, np.array, torch.Tensor, None]],
    space_modifier: Callable = lambda x: x,
    device: str = "auto",
    num_steps: int = 350,
    learning_rate: int = 3e-4,
    generator_kwargs: dict = None,
    gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
    opt_args: Dict = {},
    wrinkle_region: Union[List[int], str] = "LionWrinkle",
    work_on_copy: bool = True,
):
    if device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if generator_kwargs is None:
        generator_kwargs = {"input_is_latent": True, "return_latents": False}

    if isinstance(wrinkle_targets, np.ndarray):
        wrinkle_targets = torch.from_numpy(wrinkle_targets).to(device=device) # pylint: disable=E1101

    if work_on_copy:
        G_pti = copy.deepcopy(G).train().requires_grad_(True).to(device)
    else:
        G_pti = G.train().requires_grad_(True).to(device)

    latent_pivots = []
    for w in w_pivot:
        # w.requires_grad_(False)
        latent_pivots.append(space_modifier(w).squeeze(0))
    latent_pivots = torch.stack(latent_pivots, dim=0) # pylint: disable=E1101

    # Load LPIPS feature detector.
    lpips_loss = LpipsLoss()

    # l2 criterion
    l2_criterion = torch.nn.MSELoss(reduction='none')

    # Wrinkle features
    mask_roi = torch.from_numpy(Face.from_tensor(read_image(img_invert_path=target[0])).get_mask_landmarks( # pylint: disable=E1101
        wrinkle_region, landmark_model=LANDMARKS_MEDIAPIPE
    )).to("cuda")
    mask_roi = mask_roi.repeat(3, 1, 1)

    # Features for target image.
    targeted_images = torch.stack([ # pylint: disable=E1101
        read_image(img_invert_path=trgt, device=device).squeeze(0) 
        for trgt in target
    ], dim=0)
    c = torch.zeros([1, 0], device=device) # pylint: disable=E1101

    # initalize optimizer
    optimizer = gradient_optimizer(G_pti.parameters(), lr=learning_rate, **opt_args)


    # run optimization loop
    for _ in tqdm(range(num_steps)):

        # Synth images from opt_w.
        synth_image = G_pti(latent_pivots, c, **generator_kwargs)
        # track images
        synth_image = (synth_image.clamp(-1, 1) + 1) / 2

        # MSE loss
        mse_loss = (l2_criterion(targeted_images, synth_image)).mean()

        # Haar loss
        lpips_loss_value = lpips_loss(targeted_images, synth_image)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss = 10 * mse_loss + lpips_loss_value
        loss.backward()
        optimizer.step()

    final_img = (synth_image.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    final_img = Image.fromarray(final_img[0].cpu().numpy(), 'RGB')
    final_img.save("./final_pti.png")

    return final_img, G_pti


def pivotal_tuning(
    G: torch.nn.Module,
    w_pivot: torch.Tensor,
    target: Union[str, Image.Image, np.array, torch.Tensor],
    space_modifier: Callable = lambda x: x,
    device: str = "auto",
    num_steps: int = 350,
    learning_rate: int = 3e-4,
    generator_kwargs: dict = None,
    gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
    opt_args: Dict = {},
    lambda_low_level_features=1/100,
    progressive=False,
    negative_example=None,
):
    if device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if generator_kwargs is None:
        generator_kwargs = {"input_is_latent": True, "return_latents": False}

    # G_original = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    G_pti = copy.deepcopy(G).train().requires_grad_(True).to(device)
    pivot = space_modifier(w_pivot).detach().requires_grad_(False)

    # Load LPIPS feature detector.
    lpips_loss = LpipsLoss()

    # l2 criterion
    l2_criterion = torch.nn.MSELoss(reduction='mean')

    # Features for target image.
    target_image = read_image(img_invert_path=target, device=device)
    if negative_example is not None:
        negative_example = read_image(negative_example, device=device)

    c = torch.zeros([1, 0], device=device) # pylint: disable=E1101

    # initalize optimizer
    optimizer = gradient_optimizer(G_pti.parameters(), lr=learning_rate, **opt_args)

    if progressive:
        augment_factor = np.linspace(0, lambda_low_level_features, num_steps // 5)

    # run optimization loop
    for step in tqdm(range(num_steps)):

        if progressive and ((step + 1) % (num_steps // 5)) == 0:
            lambda_low_level_features = augment_factor[int((step + 1) // (num_steps // 5))]

        # Synth images from opt_w.
        synth_image = G_pti(pivot, c, **generator_kwargs)

        # track images
        synth_image = (synth_image.clamp(-1, 1) + 1) / 2

        # LPIPS loss
        lpips_loss_value = lpips_loss(target_image, synth_image)

        # MSE loss
        mse_loss = l2_criterion(target_image, synth_image) * 10

        # space regularizer
        # reg_loss = space_regularizer_loss(G_pti, G_original, w_pivot, lpips_loss, c, lpips_lambda=lambda_space_reg)

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss = mse_loss + lpips_loss_value
        loss.backward()
        optimizer.step()

    final_img = (synth_image.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    final_img = Image.fromarray(final_img[0].cpu().numpy(), 'RGB')
    final_img.save("./final_pti.png")

    return final_img, G_pti



class PivotalTuningEdition:

    # -------------------------------------------------------------------------
    # Overloaded Function
    # -------------------------------------------------------------------------
    def __init__(
        self, generator: torch.nn.Module, target: Union[str, Image.Image, np.array, torch.Tensor],
        device: str = "auto", generator_kwargs: dict = None, space_modifier: Callable = lambda x: x,
        wrinkle_region="LionWrinkle"
    ) -> None:
        """
        Constructor

        :param generator: Pretrained StyleGAN generator
        :type generator: torch.nn.Module
        :param target: Path or image to the target image (to invert and edit)
        :type target: Union[str, Image.Image, np.array, torch.Tensor]
        :param device: _description_, defaults to "auto"
        :type device: str, optional
        :param generator_kwargs: _description_, defaults to None
        :type generator_kwargs: dict, optional
        :param space_modifier: _description_, defaults to lambdax:x
        :type space_modifier: _type_, optional
        :param wrinkle_region: _description_, defaults to "LionWrinkle"
        :type wrinkle_region: str, optional
        """
        
        if device == "auto":
            self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self._device = device

        self._generator = generator
        self._generator_pti = None
        self._generator_kwargs = generator_kwargs

        self._target = target
        self._pivot = None
        self._space_modifier = space_modifier

        self._direction = None
        self.wrinkle_region = wrinkle_region

    # -------------------------------------------------------------------------
    # Getter / Setter
    # -------------------------------------------------------------------------

    def set_target(self, t):
        """
        

        :param t: _description_
        :type t: _type_
        """
        self._target = t


    def set_generator(self, g):
        """
        

        :param g: _description_
        :type g: _type_
        """
        self._generator = g


    def get_tuned_img(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        return self._tensor_to_pil(
            self._generator_pti(
                self._space_modifier(self._pivot),
                torch.zeros([1, 0], device=self._device), # pylint: disable=E1101
                **self._generator_kwargs
            )
        )


    def get_inverted_img(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        return self._tensor_to_pil(
            self._generator(
                self._space_modifier(self._pivot),
                torch.zeros([1, 0], device=self._device), # pylint: disable=E1101
                **self._generator_kwargs
            )
        )


    def get_pivot(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        return self._pivot


    @property
    def generator_pti(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        return self._generator_pti


    def get_direction(self):
        """


        :return: _description_
        :rtype: _type_
        """
        return self._direction


    # -------------------------------------------------------------------------
    # Saving / Loading
    # -------------------------------------------------------------------------

    def save(self, path):
        """
        

        :param path: _description_
        :type path: _type_
        """
        with open(path, "wb") as fp:
            dill.dump(self, fp)

    @staticmethod
    def load(path):
        """
        

        :param path: _description_
        :type path: _type_
        :return: _description_
        :rtype: _type_
        """
        with open(path, "rb") as fp:
            return dill.load(fp)

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    def create_pgt(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        return wrinkle_remover(path=self._target, landmarks=self.wrinkle_region)


    def invert(
        self, invertion_iteration: int = 0,
        loss_function: Callable = None,
        initial_latent_function: Callable = None,
        initial_latent_function_kwargs: Dict = None,
        loss_argues: List[Literal["syn_image", "image", "latent_space"]] = ["syn_image", "image"],
        gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        opt_args: Dict = {},
        learning_rate: float = 0.1, other_optimize_param: Dict = None
    ):
        get_latent = lambda img_h, img, lat, lat_h: lat

        self._pivot = invert_StyleGAN_experiments(
            generator=self._generator, img_invert_path=self._target,
            invertion_iteration=invertion_iteration, loss_function=loss_function,
            initial_latent_function=initial_latent_function,
            initial_latent_function_kwargs=initial_latent_function_kwargs,
            device=self._device, generator_kwargs=self._generator_kwargs,
            space_modifier=self._space_modifier, loss_argues=loss_argues,
            gradient_optimizer=gradient_optimizer, opt_args=opt_args, learning_rate=learning_rate,
            other_optimize_param=other_optimize_param,
            returned_information=get_latent
        )


    def tune(
        self, num_steps: int = 350, learning_rate: int = 3e-4,
        gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam,
        opt_args: Dict = {}, lambda_low_level_features=1/100, progressive=False,
        negative_example=None
    ):
        """
        

        :param num_steps: _description_, defaults to 350
        :type num_steps: int, optional
        :param learning_rate: _description_, defaults to 3e-4
        :type learning_rate: int, optional
        :param gradient_optimizer: _description_, defaults to torch.optim.Adam
        :type gradient_optimizer: torch.optim.Optimizer, optional
        :param opt_args: _description_, defaults to {}
        :type opt_args: Dict, optional
        :param lambda_space_reg: _description_, defaults to 0.2
        :type lambda_space_reg: float, optional
        :param features_fn: _description_, defaults to "Haar"
        :type features_fn: str, optional
        :param color_space: _description_, defaults to "RGB"
        :type color_space: str, optional
        :param lambda_low_level_features: _description_, defaults to 1/100
        :type lambda_low_level_features: _type_, optional
        :param progressive: _description_, defaults to False
        :type progressive: bool, optional
        :param negative_example: _description_, defaults to None
        :type negative_example: _type_, optional
        """
        _, self._generator_pti = pivotal_tuning(
            G=self._generator, w_pivot=self._pivot, target=self._target,
            device=self._device, num_steps=num_steps, learning_rate=learning_rate,
            generator_kwargs=self._generator_kwargs, space_modifier=self._space_modifier,
            gradient_optimizer=gradient_optimizer, opt_args=opt_args,
            lambda_low_level_features=lambda_low_level_features, progressive=progressive,
            negative_example=negative_example
        )


    def _h_stack_images(self, images: List[np.array]):
        """
        

        :param images: _description_
        :type images: List[np.array]
        :return: _description_
        :rtype: _type_
        """
        dst = Image.new('RGB', (images[0].width * len(images), images[0].height))

        for index, img in enumerate(images):
            dst.paste(img, (img.width * index, 0))

        return dst


    def _v_stack_images(self, images: List[np.array]):
        """
        

        :param images: _description_
        :type images: List[np.array]
        :return: _description_
        :rtype: _type_
        """
        dst = Image.new('RGB', (images[0].width, images[0].height * len(images)))

        for index, img in enumerate(images):
            dst.paste(img, (0, img.height * index))

        return dst


    def _tensor_to_pil(self, tensor: torch.Tensor, text: str = None):
        """
        

        :param tensor: _description_
        :type tensor: torch.Tensor
        :param text: _description_, defaults to None
        :type text: str, optional
        :return: _description_
        :rtype: _type_
        """
        img = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = Image.fromarray(img[0].cpu().numpy(), 'RGB')

        if text is not None:
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), text, (255,255,255))

        return img


    def save_exemples(self, input_gen, path, w=None):
        """
        

        :param input_gen: _description_
        :type input_gen: _type_
        :param path: _description_
        :type path: _type_
        :param w: _description_, defaults to None
        :type w: _type_, optional
        """
        pairs = []

        for i in range(5):
            latent, c = input_gen[0](), input_gen[1]()
            if w is not None:
                latent = self._space_modifier(torch.from_numpy(np.load(w[i])).to(self._device)) # pylint: disable=E1101

            synth = self._tensor_to_pil(self._generator_pti(latent, c, **self._generator_kwargs))
            synth_dir = self._tensor_to_pil(
                self._generator_pti(latent + self._direction, c, **self._generator_kwargs)
            )
            pairs += [self._h_stack_images([synth, synth_dir])]

        self._v_stack_images(pairs).save(path)


    def synth_direction(self, w, pca=None, strength=1, dry: bool = False):
        """
        

        :param w: _description_
        :type w: _type_
        :param pca: _description_, defaults to None
        :type pca: _type_, optional
        :param strength: _description_, defaults to 1
        :type strength: int, optional
        :param dry: _description_, defaults to False
        :type dry: bool, optional
        :return: _description_
        :rtype: _type_
        """

        if pca is None:
            pca = PCA(512)
            pca.fit([
                self._generator.mapping(
                    torch.from_numpy(np.random.randn(1, 512)).cuda(), # pylint: disable=E1101
                    torch.zeros([1,0], device="cuda") # pylint: disable=E1101
                ).detach().cpu().numpy()[0, 0, :] 
                for _ in range(10000)
            ])

            self._direction = [0] * 504 + np.random.randn(8).tolist()

        if not dry:
            if w.ndim == 2:
                w = w.unsqueeze(0)
            w_transform = pca.transform([w.detach().cpu().numpy()[0, 0, :]])
            w_transform += self._direction * strength
            pseudo_pivot = torch.from_numpy( # pylint: disable=E1101
                pca.inverse_transform(w_transform)
            ).cuda().requires_grad_(False)
        else:
            pseudo_pivot = None

        return pseudo_pivot, pca


    # -------------------------------------------------------------------------
    # PTI
    # -------------------------------------------------------------------------


    def synthethise(
        self,
        # Invertion params
        invertion_iteration: int = 0, 
        loss_function: Callable = None,
        initial_latent_function: Callable = None, 
        initial_latent_function_kwargs: Dict = None,
        loss_argues: List[Literal["syn_image", "image", "latent_space"]] = ["syn_image", "image"],
        gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
        opt_args: Dict = {},
        tune_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
        tune_opt_args: Dict = {},
        learning_rate_invertion: float = 0.1,
        other_optimize_param: Dict = None,
        # Tune params
        tuning_iteration: int = 1500,
        learning_rate_tuning: float = 3e-4,
        tune: bool = True,
        lambda_low_level_features=1/100,
        progressive=False,
        negative_example=None
    ):
        self.invert(
            invertion_iteration=invertion_iteration, loss_function=loss_function,
            initial_latent_function=initial_latent_function,
            initial_latent_function_kwargs=initial_latent_function_kwargs,
            loss_argues=loss_argues,
            gradient_optimizer=gradient_optimizer, opt_args=opt_args,
            learning_rate=learning_rate_invertion, other_optimize_param=other_optimize_param
        )

        if tune and tuning_iteration > 0:
            self.tune(
                num_steps=tuning_iteration, learning_rate=learning_rate_tuning,
                gradient_optimizer=tune_optimizer, opt_args=tune_opt_args,
                lambda_low_level_features=lambda_low_level_features,
                progressive=progressive, negative_example=negative_example
            )


    # -------------------------------------------------------------------------
    # PTE
    # -------------------------------------------------------------------------


    def synthethise_with_directions(
        self,
        pseudo_target: Union[str, Image.Image, np.array, torch.Tensor, None],
        target_wrinkle: Union[str, Image.Image, np.array, torch.Tensor, None] = None,
        # Invertion params
        invertion_iteration: int = 0,
        loss_function: Callable = None,
        initial_latent_function: Callable = None, 
        initial_latent_function_kwargs: Dict = None,
        loss_argues: List[Literal["syn_image", "image", "latent_space"]] = ["syn_image", "image"],
        gradient_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
        opt_args: Dict = {},
        tune_optimizer: torch.optim.Optimizer = torch.optim.Adam, 
        tune_opt_args: Dict = {},
        learning_rate_invertion: float = 0.1,
        other_optimize_param: Dict = None,
        # Tune params
        tuning_iteration: int = 1500,
        learning_rate_tuning: float = 3e-4,
        tune: bool = True,
        direction_type: Literal["Double_Invertion", "PCA", "Random", "Random_Local", "Chosen_Direction"] = "PCA",
        direction: Union[torch.Tensor, np.array] = None
    ):

        self.invert(
            invertion_iteration=invertion_iteration, loss_function=loss_function,
            initial_latent_function=initial_latent_function,
            initial_latent_function_kwargs=initial_latent_function_kwargs,
            loss_argues=loss_argues,
            gradient_optimizer=gradient_optimizer, opt_args=opt_args,
            learning_rate=learning_rate_invertion, 
            other_optimize_param=other_optimize_param
        )

        get_latent = lambda img_h, img, lat, lat_h: lat

        if direction_type == "Double_Invertion" or (direction_type == "Chosen_Direction" and direction is None):
            pseudo_pivot = invert_StyleGAN_experiments(
                generator=self._generator, img_invert_path=self._target,
                invertion_iteration=100, loss_function=loss_function,
                initial_latent_function=self._pivot.clone().detach(),
                device=self._device, generator_kwargs=self._generator_kwargs,
                space_modifier=self._space_modifier, loss_argues=loss_argues,
                gradient_optimizer=gradient_optimizer, opt_args=opt_args,
                learning_rate=learning_rate_invertion,
                other_optimize_param=other_optimize_param,
                returned_information=get_latent
            )
            self._direction = self._pivot - pseudo_pivot
        elif direction_type == "Random_Local":
            self._direction = torch.rand_like(self._pivot, device=self._pivot.device) - 0.5  # pylint: disable=E1101
            self._direction *= (5 * self._direction.size()[1]) / torch.abs(self._direction).sum()  # pylint: disable=E1101
            self._direction = self._direction.detach().requires_grad_(False)
            pseudo_pivot = self._pivot - self._direction
        elif direction_type == "Random_Local":
            self._direction = torch.rand_like(self._pivot, device=self._pivot.device)  # pylint: disable=E1101
            self._direction = self._direction.detach().requires_grad_(False)
            pseudo_pivot = self._pivot - self._direction
        elif direction_type == "PCA":
            pca = PCA(512)
            pca.fit([
                self._generator.mapping(
                    torch.from_numpy(np.random.randn(1, 512)).cuda(), # pylint: disable=E1101
                    torch.zeros([1,0], device="cuda")  # pylint: disable=E1101
                ).detach().cpu().numpy()[0, 0, :]
                for _ in range(10000)
            ])

            w_transform = pca.transform([self._pivot.detach().cpu().numpy()[0, 0, :]])
            w_transform += [0] * 504 + np.random.randn(8).tolist()
            pseudo_pivot = torch.from_numpy( # pylint: disable=E1101
                pca.inverse_transform(w_transform)
            ).cuda().requires_grad_(False)

            self._direction = self._pivot - pseudo_pivot
        elif direction_type == "Chosen_Direction":
            self._direction = direction
            pseudo_pivot = self._pivot - self._direction


        pivots = [
            self._pivot, pseudo_pivot
        ]

        targets = [
            self._target, pseudo_target
        ]

        if tune and tuning_iteration > 0:
            _, self._generator_pti = pivotal_tuning_direction(
                G=self._generator, w_pivot=pivots, target=targets,
                device=self._device, num_steps=tuning_iteration,
                learning_rate=learning_rate_tuning,
                generator_kwargs=self._generator_kwargs, 
                space_modifier=self._space_modifier,
                gradient_optimizer=tune_optimizer, opt_args=tune_opt_args, 
                wrinkle_region=self.wrinkle_region, wrinkle_targets=target_wrinkle
            )


    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def edit_pivot(self, alpha):
        """
        

        :param alpha: _description_
        :type alpha: _type_
        :return: _description_
        :rtype: _type_
        """
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.Tensor([alpha]).cuda()

        return self._tensor_to_pil(
            self._generator_pti(
                self._space_modifier(self._pivot + torch.sum( # pylint: disable=E1101
                    alpha[:, None, None].repeat(1, 1, 512) * self._direction, dim=0
                ).unsqueeze(0)), torch.zeros(1, 0, device="cuda"), **self._generator_kwargs, # pylint: disable=E1101
            )
        )

    def mse_full(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        original_image = cv2.imread(self._target) # pylint: disable=E1101

        c = torch.zeros([1, 0], device=self._device) # pylint: disable=E1101
        synth = self._generator_pti(self._space_modifier(self._pivot), c, **self._generator_kwargs)
        synth = cv2.cvtColor(  # pylint: disable=E1101
            (synth.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0],
            cv2.COLOR_RGB2BGR  # pylint: disable=E1101
        )

        return np.sqrt(np.square(original_image - synth).mean())


    def identity_loss(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        original_image = Image.open(self._target)

        c = torch.zeros([1, 0], device=self._device) # pylint: disable=E1101
        synth_image = self._tensor_to_pil(self._generator_pti(self._space_modifier(self._pivot), c, **self._generator_kwargs))

        mtcnn = facenet_pytorch.MTCNN(image_size=original_image.size[0])
        crop_original_image = mtcnn(original_image)
        crop_synth_image = mtcnn(synth_image)

        resnet = facenet_pytorch.InceptionResnetV1(pretrained='vggface2').eval().to("cuda")
        original_embedding = resnet(crop_original_image.unsqueeze(0).to("cuda")) # pylint: disable=E1102
        synth_embedding = resnet(crop_synth_image.unsqueeze(0).to("cuda")) # pylint: disable=E1102

        return torch.nn.functional.mse_loss(original_embedding, synth_embedding).detach().item()


    def global_variations(self):
        """
        

        :return: _description_
        :rtype: _type_
        """
        c = torch.zeros([1, 0], device=self._device) # pylint: disable=E1101

        synth_dir1 = self._generator_pti(
            self._space_modifier(self._pivot + self._direction), c, **self._generator_kwargs
        )
        synth_dir1 = cv2.cvtColor( # pylint: disable=E1101
            (synth_dir1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0],
            cv2.COLOR_RGB2BGR # pylint: disable=E1101
        )

        synth_dir2 = self._generator_pti(self._space_modifier(self._pivot - self._direction), c, **self._generator_kwargs)
        synth_dir2 = cv2.cvtColor( # pylint: disable=E1101
            (synth_dir2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0],
            cv2.COLOR_RGB2BGR # pylint: disable=E1101
        )

        return np.sqrt(np.square(synth_dir1 - synth_dir2).mean())


    def show_result(self, save_path: str = None, only_inverted: bool = False):
        """
        

        :param save_path: _description_, defaults to None
        :type save_path: str, optional
        :param only_inverted: _description_, defaults to False
        :type only_inverted: bool, optional
        :return: _description_
        :rtype: _type_
        """

        if isinstance(self._target, str):
            original_image = Image.open(self._target)
        else:
            original_image = self._target

        c = torch.zeros([1, 0], device=self._device) # pylint: disable=E1101
        inverted_image = self._tensor_to_pil(
            self._generator(self._space_modifier(self._pivot), c, **self._generator_kwargs)
        )

        if only_inverted:
            img = inverted_image
        elif self._generator_pti is not None:
            tuned_image = self._tensor_to_pil(
                self._generator_pti(
                    self._space_modifier(self._pivot), c, **self._generator_kwargs
                ), "PTI"
            )

            img = self._h_stack_images([original_image, inverted_image, tuned_image])
        else:
            img = self._h_stack_images([original_image, inverted_image])

        if save_path is not None:
            img.save(save_path)

        return img
