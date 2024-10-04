bl_info = {
    "name": "Character Generator with FLUX",
    "blender": (3, 0, 0),
    "category": "3D View",
}

import bpy
import os
import subprocess
import sys
from PIL import Image, ImageFilter
import math

#Fix: asset and file name will be overwritten.

class FLUX_OT_SetupEnvironment(bpy.types.Operator):
    """Set up a virtual environment and install dependencies"""
    bl_idname = "object.setup_flux_env"
    bl_label = "Set up Environment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            # Get the current Blender Python executable
            python_executable = sys.executable

            # Path for virtual environment
            venv_dir = bpy.path.abspath("//flux_venv")

            # Step 1: Create the virtual environment
            if not os.path.exists(venv_dir):
                subprocess.run([python_executable, "-m", "venv", venv_dir], check=True)
                self.report({'INFO'}, f"Virtual environment created at {venv_dir}")
            else:
                self.report({'INFO'}, "Virtual environment already exists.")

            # Step 2: Install dependencies
            self.install_dependencies(venv_dir)

            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error setting up environment: {str(e)}")
            return {'CANCELLED'}

    def install_dependencies(self, venv_dir):
        """Install required Python packages in the virtual environment"""
        python_executable = os.path.join(venv_dir, "bin", "python")  # Linux/Unix path to python
        if sys.platform == "win32":
            python_executable = os.path.join(venv_dir, "Scripts", "python.exe")  # Windows path to python

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
                #"--user",
                "--upgrade",
            ]
        )

        # Packages to install
        packages = [
            "diffusers",
            "transformers",
            "Pillow",
        ]

        # Install packages using the virtual environment's pip
        for package in packages:
            subprocess.run(
                [python_executable, "-m", "pip", "install", package, "--upgrade"],
                check=True
            )

        self.report({'INFO'}, "Dependencies installed successfully.")



def get_unique_asset_name(self, context):
    """Generates a unique asset name if there is a conflict."""
    base_name = context.scene.asset_name
    existing_names = {obj.name for obj in bpy.data.objects if obj.asset_data}
    
    # If no conflict, return the original name
    if base_name not in existing_names:
        return base_name
    
    # Add suffix like (1), (2), ... until a unique name is found
    counter = 1
    unique_name = f"{base_name} ({counter})"
    while unique_name in existing_names:
        counter += 1
        unique_name = f"{base_name} ({counter})"
        
    context.scene.asset_name = unique_name
    
    return

def get_unique_file_name(base_path):
    """Generates a unique file name if there is a conflict in the file system."""
    # Split the file name and extension
    base_name, extension = os.path.splitext(base_path)
    
    # If no conflict, return the original path
    if not os.path.exists(base_path):
        return base_path
    
    # Add suffix like (1), (2), ... until a unique file name is found
    counter = 1
    unique_path = f"{base_name} ({counter}){extension}"
    while os.path.exists(unique_path):
        counter += 1
        unique_path = f"{base_name} ({counter}){extension}"
    
    return unique_path


class FLUX_OT_GenerateCharacter(bpy.types.Operator):
    """Generate character image from description and convert to 3D object"""
    bl_idname = "object.generate_character"
    bl_label = "Generate Character"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            # Fetch the character description from the scene
            description = context.scene.character_description

            # Ensure the description is not empty
            if not description:
                self.report({'ERROR'}, "Character description is empty.")
                return {'CANCELLED'}

            # Generate image using FLUX
            image_path = self.generate_image(context, description)

            # Remove background from the generated image
            transparent_image_path = self.remove_background(context, image_path)

            # Convert the transparent image to a 3D object
            self.convert_to_3d(context, transparent_image_path, description)

            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}

    def generate_image(self, context, description):
        """Generates an image using the FLUX model based on the user input."""
        # Import dependencies inside the method to avoid potential module issues before installation
        from diffusers import FluxPipeline
        import torch
        asset_name = context.scene.asset_name

        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        #pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.vae.enable_tiling()
        # Generate the image
        prompt = "wide-shot, full body shot, neutral background, " + description
        out = pipe(
            prompt=prompt,
            guidance_scale=0.,
            height=768,
            width=1360,
            num_inference_steps=4,
            max_sequence_length=256,
        ).images[0]

        # Save the generated image
        image_path = bpy.path.abspath(f"//{context.scene.asset_name}_generated_image.png")
        print(image_path)
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

        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

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
        transparent_image_path = bpy.path.abspath(f"//{context.scene.asset_name}_generated_image_transparent.png")  # Use asset name
        print(transparent_image_path)
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
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

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
        asset_name = context.scene.asset_name

        # Ensure the image exists
        if not os.path.exists(transparent_image_path):
            self.report({'ERROR'}, f"Image not found at {transparent_image_path}")
            return

        # Load the image into Blender
        image = image = Image.open(transparent_image_path).convert("RGB")
        #image = bpy.data.images.load(transparent_image_path)

       # Create a mask and crop the image to non-transparent areas
        processed_image = self.process_image(image)
        
        # Save the cropped image
        processed_image_path = bpy.path.abspath("//"+asset_name+"_processed_image.png")
        processed_image.save(processed_image_path)

        # Create a new material with transparency support
        material = bpy.data.materials.new(name="ImageMaterial")
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs[12].default_value = 0

        # Load the image into the material's base color and alpha inputs
        tex_image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        tex_image_node.image = bpy.data.images.load(processed_image_path)
        tex_image_node.interpolation = 'Linear'
        
        # Connect the texture's color and alpha channels to the material's shader
        material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image_node.outputs['Color'])
        material.node_tree.links.new(bsdf.inputs['Alpha'], tex_image_node.outputs['Alpha'])

        # Enable transparency in the material
        material.blend_method = 'BLEND'
        material.shadow_method = 'HASHED'

        # Create a plane to hold the image
        bpy.ops.mesh.primitive_plane_add(size=1)
        obj = bpy.context.object

        # Set the name for the new plane
        obj.name = asset_name

        # Assign the material to the plane
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

        # Adjust the plane's size to match the image aspect ratio
        img_width, img_height = processed_image.size
        aspect_ratio = img_width / img_height
        obj.scale = (aspect_ratio, 1, 1)

        # Apply transforms
        bpy.ops.transform.rotate(value=math.radians(-90), orient_axis='X')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # Mark the object as an asset
        obj.asset_mark()
        
        with context.temp_override(id=obj):  
            bpy.ops.ed.lib_id_load_custom_preview(filepath=transparent_image_path)

        # Set asset metadata
        obj.asset_data.author = "2D People Add-on"
        obj.asset_data.description = prompt #f"Generated from: {os.path.basename(transparent_image_path)}"
        obj.asset_data.tags.new(name="GeneratedCharacter")

        # Save the .blend file so that the asset is persistent
        bpy.ops.wm.save_mainfile()
        
        self.report({'INFO'}, "3D object created and added to the asset library")

# UI Panel for Character Generation and Setup
class FLUX_PT_GenerateCharacterPanel(bpy.types.Panel):
    """Creates a Panel in the 3D View for the FLUX character generator"""
    bl_label = "2D People Generator"
    bl_idname = "VIEW3D_PT_generate_character"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = '2D People'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Button to install dependencies
        layout.operator("object.setup_flux_env", text="Set up Environment")

        # Button to generate the character
        layout.prop(scene, "character_description", text="Prompt")
        layout.prop(context.scene, "asset_name", text="Name")
        layout.operator("object.generate_character",text="Generate")


# Registering the add-on and properties
def register():
    bpy.utils.register_class(FLUX_OT_SetupEnvironment)
    bpy.utils.register_class(FLUX_OT_GenerateCharacter)
    bpy.utils.register_class(FLUX_PT_GenerateCharacterPanel)

    # Register the character_description as a scene property
    bpy.types.Scene.character_description = bpy.props.StringProperty(
        name="Character Description",
        description="Describe the character to generate",
        default=""
    )
    bpy.types.Scene.asset_name = bpy.props.StringProperty(  # Add asset name property
        name="Asset Name",
        description="Name for the generated asset",
        default="",
        update=get_unique_asset_name,
    )    
    

def unregister():
    # Unregister classes
    bpy.utils.unregister_class(FLUX_OT_SetupEnvironment)
    bpy.utils.unregister_class(FLUX_OT_GenerateCharacter)
    bpy.utils.unregister_class(FLUX_PT_GenerateCharacterPanel)

    # Remove the character_description property
    del bpy.types.Scene.character_description
    del bpy.types.Scene.asset_name

if __name__ == "__main__":
    register()
    #unregister()
