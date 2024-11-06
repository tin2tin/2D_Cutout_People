bl_info = {
    "name": "2D Asset Generator",
    "author": "tintwotin",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "category": "3D View",
    "location": "3D Editor > Sidebar > 2D Asset",
    "description": "2D Asset Generator in the 3D View",
}

import bpy
from bpy.types import Operator, PropertyGroup, Panel
from bpy.props import StringProperty, EnumProperty
import os, re
import subprocess
import sys
from PIL import Image, ImageFilter
import math
from os.path import join
from mathutils import Vector


dir_path = os.path.join(bpy.utils.user_resource("DATAFILES"), "2D Assets")
os.makedirs(dir_path, exist_ok=True)


def flush():
    import torch
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    # torch.cuda.reset_peak_memory_stats()


def python_exec():
    return sys.executable


# Get a list of text blocks in Blender
def texts(self, context):
    return [(text.name, text.name, "") for text in bpy.data.texts]


# Property Group for storing the selected text block and toggle
class Import_Text_Props(PropertyGroup):
    def update_text_list(self, context):
        self.script = bpy.data.texts[self.scene_texts].name
        return None

    # EnumProperty to toggle between Text-Block and Prompt
    input_type: EnumProperty(
        name="Input Type",
        description="Choose between Text-Block and Prompt",
        items=[
            ('PROMPT', "Prompt", "Use Prompt"),
            ('TEXT_BLOCK', "Text-Block", "Use Text Block"),
        ],
        default='TEXT_BLOCK',
    )

    script: StringProperty(default="", description="Browse Text to be Linked")
    scene_texts: EnumProperty(
        name="Text-Blocks",
        items=texts,
        update=update_text_list,
        description="Text-Blocks",
    )


class FLUX_OT_SetupEnvironment(bpy.types.Operator):
    """Set up a environment and install dependencies"""

    bl_idname = "object.setup_flux_env"
    bl_label = "Set up Environment"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # try:
        # import sys
        # Get the current Blender Python executable
        # python_executable = sys.executable

        #            # Path for virtual environment
        #            venv_dir = bpy.path.abspath("//flux_venv")

        #            # Step 1: Create the virtual environment
        #            if not os.path.exists(venv_dir):
        #                subprocess.run([python_executable, "-m", "venv", venv_dir], check=True)
        #                self.report({'INFO'}, f"Virtual environment created at {venv_dir}")
        #            else:
        #                self.report({'INFO'}, "Virtual environment already exists.")
        # Step 2: Install dependencies
        self.install_dependencies(python_exec())

        return {"FINISHED"}

    #        except Exception as e:
    #            self.report({'ERROR'}, f"Error setting up environment: {str(e)}")
    #            return {'CANCELLED'}

    def install_dependencies(self, venv_dir):
        """Install required Python packages in the virtual environment"""
        python_executable = venv_dir  # os.path.join(venv_dir, "bin", "python")  # Linux/Unix path to python
        #        if sys.platform == "win32":
        #            python_executable = os.path.join(venv_dir, "Scripts", "python.exe")  # Windows path to python
        subprocess.check_call(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "torch==2.3.1+cu121",
                "xformers",
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "--no-warn-script-location",
                # "--user",
                "--upgrade",
            ]
        )

        # Packages to install
        packages = [
            "diffusers",
            "transformers",
            "Pillow",
            "bitsandbytes",
            "botocore",
            "ml-dtypes",
            "protobuf==3.20.1",
            "tqdm",
            "markupsafe",
        ]

        # Install packages using the virtual environment's pip
        for package in packages:
            subprocess.run(
                [
                    python_executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--use-deprecated=legacy-resolver",
                    package,
                    "--no-warn-script-location",
                    "--upgrade",
                ],
                check=True,
            )
        self.report({"INFO"}, "\nDependencies installed successfully.")


def get_unique_asset_name(self, context):
    """Generates a unique asset name if there is a conflict, ensuring a name is always returned."""

    # Retrieve base name and check for validity
    base_name = context.scene.asset_name
    if not base_name:
        # If base name is missing, use the asset prompt or a default
        prompt = context.scene.asset_prompt
        base_name = "_".join(prompt.split()[:2]) if prompt else "Asset"
        context.scene.asset_name = base_name

    # Collect existing names to detect conflicts
    existing_names = {obj.name for obj in bpy.data.objects if getattr(obj, "asset_data", None)}

    # If the base name is unique, return it directly
    if base_name not in existing_names:
        return

    # Attempt to extract an existing number suffix in parentheses, if present
    match = re.search(r"\((\d+)\)$", base_name)
    if match:
        base_name = base_name[: match.start()].strip()
        counter = int(match.group(1)) + 1
    else:
        counter = 1

    # Generate a unique name by incrementing the counter until no conflicts remain
    unique_name = f"{base_name} ({counter})"
    while unique_name in existing_names:
        counter += 1
        unique_name = f"{base_name} ({counter})"

    # Set the unique name in the context and return it
    context.scene.asset_name = unique_name
    return


def get_unique_file_name(base_path):
    """Generates a unique file name if there is a conflict in the file system."""
    base_name, extension = os.path.splitext(base_path)

    # Regular expression to detect if the file name has a number in parentheses
    match = re.search(r"\((\d+)\)$", base_name)

    if match:
        # If there's a number, increment it
        base_name = base_name[: match.start()].strip()
        counter = int(match.group(1)) + 1
    else:
        # If no number, start at 1
        counter = 1
    unique_path = f"{base_name} ({counter}){extension}"
    while os.path.exists(unique_path):
        counter += 1
        unique_path = f"{base_name} ({counter}){extension}"
    return unique_path


class FLUX_OT_GenerateAsset(bpy.types.Operator):
    """Generate asset image from description and convert to 3D object"""

    bl_idname = "object.generate_asset"
    bl_label = "Generate Asset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            pipe = self.load_model(context)
            input_type = context.scene.import_text.input_type

            if input_type == 'TEXT_BLOCK':
                # text = bpy.data.texts[import_text.scene_texts]
                text = bpy.data.texts[context.scene.import_text.scene_texts]
                lines = [line.body for line in text.lines]
                lines = [line for line in lines if line.strip()]
            elif input_type == 'PROMPT':
                lines = [context.scene.asset_prompt]

            for index, line in enumerate(lines):
                if line:
                    # Fetch the character description from the scene
                    context.scene.asset_prompt = line
                    base_name = " ".join(line.split()[:3]) if line else "Asset"
                    context.scene.asset_name = base_name.title()
                    description = context.scene.asset_prompt
                    print(str(index + 1) + "/" + str(len(lines)) + ": " + base_name.title())

                    # Ensure the description is not empty
                    if not description:
                        self.report({"ERROR"}, "Asset description is empty.")
                        return {"CANCELLED"}

                    # Generate image using FLUX
                    image_path = bpy.path.abspath(self.generate_image(context, description, pipe))
                    print(image_path)

                    # Remove background from the generated image
                    transparent_image_path = bpy.path.abspath(self.remove_background(context, image_path))
                    print(transparent_image_path)

                    # Convert the transparent image to a 3D object
                    self.convert_to_3d(context, transparent_image_path, description)
            flush()
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Error: {str(e)}")
            return {"CANCELLED"}

#    #SD 3.5 Medium
#    def generate_image(self, context, description):
#        """Generates an image using the Stable Diffusion 3 model based on user input."""

#        # Import dependencies inside the method to avoid potential module issues before installation
#        from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
#        import torch

#        # Define model configuration and ID
#        model_id = "stabilityai/stable-diffusion-3.5-medium"
#        asset_name = context.scene.asset_name

#        # Configure quantization settings for 4-bit loading
#        nf4_config = BitsAndBytesConfig(
#            load_in_4bit=True,
#            bnb_4bit_quant_type="nf4",
#            bnb_4bit_compute_dtype=torch.bfloat16
#        )

#        # Initialize the transformer model with quantization settings
#        model_nf4 = SD3Transformer2DModel.from_pretrained(
#            model_id,
#            subfolder="transformer",
#            quantization_config=nf4_config,
#            torch_dtype=torch.bfloat16
#        )

#        # Load the Stable Diffusion pipeline with the transformer model
#        pipeline = StableDiffusion3Pipeline.from_pretrained(
#            model_id,
#            transformer=model_nf4,
#            torch_dtype=torch.bfloat16
#        )

#        # Enable CPU offloading for memory optimization
#        pipeline.enable_model_cpu_offload()

#        # Construct the prompt and generate the image
#        prompt = "neutral background, " + description
#        out = pipeline(
#            prompt=prompt,
#            guidance_scale=2.8,
#            height=1440,
#            width=1440,
#            num_inference_steps=30,
#            max_sequence_length=256,
#        ).images[0]

#        # Save the generated image to the specified path
#        asset_name = re.sub(r'[<>:"/\\|?*]', '', context.scene.asset_name)
#        image_path = bpy.path.abspath(f"//{asset_name}_generated_image.png")
#        out.save(image_path)
#        flush()
#        return image_path


    # FLUX
    def load_model(self, context):
        """Generates an image using the FLUX model based on the user input."""

        # Import dependencies inside the method to avoid potential module issues before installation
        from diffusers import FluxPipeline
        import torch

        #asset_name = context.scene.asset_name
        # If bitsandbytes doesn't work, use this:
        #        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        #        pipe.enable_sequential_cpu_offload()
        #        pipe.enable_vae_slicing()
        #        pipe.vae.enable_tiling()

        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

        image_model_card = "ChuckMcSneed/FLUX.1-dev"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_nf4 = FluxTransformer2DModel.from_pretrained(
            image_model_card,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
        )

        pipe = FluxPipeline.from_pretrained(image_model_card, transformer=model_nf4, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe

    # FLUX
    def generate_image(self, context, description, pipe):
        """Generates an image using the FLUX model based on the user input."""

        asset_name = context.scene.asset_name

        # Generate the image
        prompt = "neutral background, " + description
        out = pipe(
            prompt=prompt,
            guidance_scale=3.8,
            height=1024,
            width=1024,
            num_inference_steps=25,
            max_sequence_length=256,
        ).images[0]

        # Save the generated image
        asset_name = re.sub(r'[<>:"/\\|?*]', '', context.scene.asset_name)
        image_path = bpy.path.abspath(
            join(
                bpy.utils.user_resource("DATAFILES"),
                "2D Assets",
                f"{asset_name}_generated_image.png",
            )
        )
        out.save(image_path)
        return image_path

    def remove_background(self, context, image_path):
        """Removes the background from the image using the BiRefNet segmentation model."""
        # Import dependencies inside the method
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        from PIL import Image, ImageFilter
        import torch

        asset_name = context.scene.asset_name

        birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        birefnet.to("cuda")

        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Load and transform the image
        image = Image.open(image_path).convert("RGB")
        image_size = image.size
        input_image = transform_image(image).unsqueeze(0).to("cuda")

        # Generate the background mask
        with torch.no_grad():
            preds = birefnet(input_image)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred)
        mask = mask.resize(image_size)

        # Refine the mask: Apply thresholding and feathering for smoother removal
        refined_mask = self.refine_mask(mask)

        # Apply the refined mask to the image to remove the background
        image.putalpha(refined_mask)
        asset_name = re.sub(r'[<>:"/\\|?*]', '', context.scene.asset_name)
        transparent_image_path = bpy.path.abspath(
            join(
                bpy.utils.user_resource("DATAFILES"),
                "2D Assets",
                f"{asset_name}_generated_image_transparent.png",
            )
        )  # Use asset name
        image.save(transparent_image_path)

        return transparent_image_path

    def refine_mask(self, mask):
        """Refines the mask by applying thresholding and feathering."""
        mask = mask.convert("L")

        # Apply thresholding
        threshold_value = 200
        mask = mask.point(lambda p: 255 if p > threshold_value else 0)

        # Apply feathering (blur)
        feather_radius = 1
        mask = mask.filter(ImageFilter.GaussianBlur(feather_radius))

        return mask

    def process_image(self, image):
        """Process the image for background removal and crop to the non-transparent areas."""
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        birefnet.to("cuda")
        image_size = image.size
        transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_images = transform_image(image).unsqueeze(0).to("cuda")

        # Prediction
        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)

        # Create a mask from the prediction
        mask = pred_pil.resize(image_size)

        # Apply the mask to the original image
        image.putalpha(mask)

        # Crop the image to the non-transparent areas
        return self.crop_to_non_transparent(image)

    def crop_to_non_transparent(self, image):
        """Crops the image to the bounding box of non-transparent areas."""

        # Convert to RGBA if not already
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Get the data from the image
        data = image.getdata()

        # Create a mask for the non-transparent pixels
        non_transparent_pixels = [(r, g, b, a) for r, g, b, a in data if a > 0]

        # If there are no non-transparent pixels, return the original image
        if not non_transparent_pixels:
            return image

        # Find the bounding box of non-transparent pixels
        x_coords = [i % image.width for i in range(len(data)) if data[i][3] > 0]
        y_coords = [i // image.width for i in range(len(data)) if data[i][3] > 0]

        left = min(x_coords)
        right = max(x_coords)
        top = min(y_coords)
        bottom = max(y_coords)

        # Crop the image to the bounding box
        return image.crop((left, top, right + 1, bottom + 1))

    def convert_to_3d(self, context, transparent_image_path, prompt):
        """Converts an image with transparency into a 3D object (plane) and adds it to the asset library."""
        import os
        import bpy

        get_unique_asset_name(self, context)
        asset_name = context.scene.asset_name

        # Ensure the image exists
        if not os.path.exists(transparent_image_path):
            self.report({"ERROR"}, f"Image not found at {transparent_image_path}")
            return {"CANCELLED"}

        # Load the image into Blender
        image = image = Image.open(transparent_image_path).convert("RGB")

        # Create a mask and crop the image to non-transparent areas
        processed_image = self.process_image(image)
        asset_name = re.sub(r'[<>:"/\\|?*]', '', asset_name)
        # Save the cropped image
        processed_image_path = bpy.path.abspath(
            os.path.join(
                bpy.utils.user_resource("DATAFILES"),
                "2D Assets",
                asset_name + "_processed_image.png",
            )
        )
        processed_image.save(processed_image_path)
        print(processed_image_path)

        # Create a new material with transparency support
        material = bpy.data.materials.new(name="ImageMaterial")
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs[12].default_value = 0  # Set Alpha to 0 for transparency
        bsdf.inputs["IOR"].default_value = 1.0  # Minimum effective IOR for transparency

        # Load the image into the material's base color and alpha inputs
        tex_image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        tex_image_node.image = bpy.data.images.load(processed_image_path)
        tex_image_node.interpolation = "Linear"

        # Add a ColorRamp node between the texture and BSDF
        color_ramp_node = material.node_tree.nodes.new("ShaderNodeValToRGB")

        # Position nodes for better visual organization in the node editor
        tex_image_node.location = (-900, 300)
        color_ramp_node.location = (-600, 300)
        bsdf.location = (-300, 300)

        # Connect the texture's color output to the ColorRamp node input
        material.node_tree.links.new(color_ramp_node.inputs["Fac"], tex_image_node.outputs["Alpha"])

        # Connect the ColorRamp output to the Base Color input of the Principled BSDF
        material.node_tree.links.new(bsdf.inputs["Alpha"], color_ramp_node.outputs["Color"])

        # Connect the texture's alpha output to the BSDF's alpha input
        material.node_tree.links.new(bsdf.inputs["Base Color"], tex_image_node.outputs["Color"])

        # Adjust ColorRamp black and white stops
        color_ramp_node.color_ramp.elements[0].position = 0.75  # Move black point to 75%
        color_ramp_node.color_ramp.elements[0].color = (
            0,
            0,
            0,
            1,
        )  # Ensure black is fully black

        color_ramp_node.color_ramp.elements[1].position = 0.95  # Move white point to 95%
        color_ramp_node.color_ramp.elements[1].color = (
            1,
            1,
            1,
            1,
        )  # Ensure white is fully white

        #        # Create a new material with transparency support
        #        material = bpy.data.materials.new(name="ImageMaterial")
        #        material.use_nodes = True
        #        bsdf = material.node_tree.nodes.get("Principled BSDF")
        #        bsdf.inputs[12].default_value = 0

        #        # Load the image into the material's base color and alpha inputs
        #        tex_image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        #        tex_image_node.image = bpy.data.images.load(processed_image_path)
        #        tex_image_node.interpolation = 'Linear'

        #        # Connect the texture's color and alpha channels to the material's shader
        #        material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
        #        material.node_tree.links.new(bsdf.inputs['Alpha'], tex_image_node.outputs['Alpha'])

        # Enable transparency in the material
        material.blend_method = "HASHED"  #'BLEND'
        material.shadow_method = "OPAQUE"
        material.surface_render_method = "DITHERED"

        # obj = bpy.ops.image.import_as_mesh_planes(filepath=transparent_image_path, files=[{"name":os.path.basename(transparent_image_path), "name":os.path.basename(transparent_image_path)}], directory=os.path.dirname(transparent_image_path))

        # Create a plane to hold the image
        bpy.ops.mesh.primitive_plane_add(size=1, location=bpy.context.scene.cursor.location)
        obj = bpy.context.object

        # Set the name for the new plane
        obj.name = asset_name

        # Assign the material to the plane
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        # obj.active_material.surface_render_method = 'BLENDED'

        # Adjust the plane's size to match the image aspect ratio
        img_width, img_height = processed_image.size
        aspect_ratio = img_width / img_height
        obj.scale = (aspect_ratio, 1, 1)

        # Apply transforms
        bpy.ops.transform.rotate(value=math.radians(-90), orient_axis="X")
        bpy.ops.transform.translate(
            value=(0, 0, 0.5),
            orient_type="GLOBAL",
            constraint_axis=(False, False, True),
        )
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Set origin point
        #bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

        bpy.context.view_layer.objects.active = obj
        
        # Avoid deleting the asset when deleting the object
        obj.data.use_fake_user = True

        # Mark the object as an asset
        obj.asset_mark()

        with context.temp_override(id=obj):
            bpy.ops.ed.lib_id_load_custom_preview(filepath=transparent_image_path)

        # Set asset metadata
        obj.asset_data.author = "2D Asset Generator"
        obj.asset_data.description = prompt  # f"Generated from: {os.path.basename(transparent_image_path)}"
        obj.asset_data.tags.new(name="GeneratedAsset")

        # Save the .blend file so that the asset is persistent
        bpy.ops.wm.save_mainfile()

        get_unique_asset_name(self, context)

        self.report({"INFO"}, "3D object created and added to the asset library")


# UI Panel for Asset Generation and Setup
class FLUX_PT_GenerateAssetPanel(bpy.types.Panel):
    """Creates a Panel in the 3D View for the FLUX character generator"""

    bl_label = "2D Asset Generator"
    bl_idname = "VIEW3D_PT_generate_asset"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "2D Asset"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        import_text = scene.import_text

        # Button to install dependencies
        layout = self.layout
        layout.operator("object.setup_flux_env", text="Set-up Dependencies")

        layout = layout.box()

        # Toggle between Text-Block and Prompt as an expandable row
        row = layout.row()
        row.prop(import_text, "input_type", expand=True)

        # Show Text-Block selector if 'TEXT_BLOCK' is selected, otherwise show the prompt input
        if import_text.input_type == 'TEXT_BLOCK':
            row = layout.row(align=True)
            row.prop(import_text, "scene_texts", text="", icon="TEXT", icon_only=True)
            row.prop(import_text, "script", text="")
        else:
            layout.prop(scene, "asset_prompt", text="Prompt")

        # Button to generate the character
        layout.prop(context.scene, "asset_name", text="Name")
        
        layout.operator("object.generate_asset", text="Generate")        


# Register and Unregister classes and properties
classes = (
    Import_Text_Props,
    FLUX_OT_SetupEnvironment,
    FLUX_OT_GenerateAsset,
    FLUX_PT_GenerateAssetPanel,
)


# Registering the add-on and properties
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.import_text = bpy.props.PointerProperty(type=Import_Text_Props)

    # Register the asset_prompt as a scene property
    bpy.types.Scene.asset_prompt = bpy.props.StringProperty(
        name="Asset Description",
        description="Describe the asset to generate",
        default="",
    )
    bpy.types.Scene.asset_name = bpy.props.StringProperty(  # Add asset name property
        name="Asset Name",
        description="Name for the generated asset",
        default="",
        update=get_unique_asset_name,
    )


def unregister():
    del bpy.types.Scene.import_text
    for cls in classes:
        bpy.utils.unregister_class(cls)

    # Remove the asset_prompt property
    del bpy.types.Scene.asset_prompt
    del bpy.types.Scene.asset_name


if __name__ == "__main__":
    register()
