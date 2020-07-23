import matplotlib.pyplot as plt
def plot_test_generated_images_for_model(output_dir, generator, x_test_lr, x_test_hr , dim=(1, 3), figsize=(15, 5)):
    
    '''
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)
    '''
    examples = x_test_hr.shape[0]
    image_batch_hr = x_test_hr
    image_batch_lr = x_test_lr
    generated_image = generator.predict(image_batch_lr)
    

    
    for index in range(examples):
    
        plt.figure(figsize=figsize)
    
        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[index][:,:,[7,15,21]])#, interpolation='nearest')
        plt.axis('off')
        
        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[index][:,:,[7,15,21]])#, interpolation='nearest')
        plt.axis('off')
    
        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[index][:,:,[7,15,21]])#, interpolation='nearest')
        plt.axis('off')
    
        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.png' % index)
        
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


#Test
def test_model(model,output_dir, x_test_lr, x_test_hr):

  plot_test_generated_images_for_model(output_dir, model, x_test_lr, x_test_hr)