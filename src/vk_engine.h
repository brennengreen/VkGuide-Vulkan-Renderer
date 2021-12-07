// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include <vk_mesh.h>

struct PipelineBuilder {
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;

	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

struct DeletionQueue {
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) (*it)();
		deletors.clear();
	}
};

class VulkanEngine {
public:

	// Cleanup
	DeletionQueue _mainDeletionQueue;

	// Vulkan Initilization Members
	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	// Swapchain Members
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	// Vulkan Commands
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	// Render Passes
	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers;

	// Synchronization
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	// Graphics Pipeline
	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;
	VkPipeline _redTrianglePipeline;
	VkPipeline _meshPipeline;
	Mesh _triangleMesh;

	// Shaders
	int _selectedShader { 0 };

	// Memory Allocater
	VmaAllocator _allocator;

public:

	bool _isInitialized{ false };
	int _frameNumber {0};

	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

private:

	// Init
	void _init_vulkan();
	void _init_swapchain();
	void _init_commands();
	void _init_default_renderpass();
	void _init_framebuffers();
	void _init_sync_structures();
	void _init_pipelines();
	
	// Shaders
	bool _load_shader_module(const char* filePath, VkShaderModule* outShaderModule);

	// Meshes
	void _load_meshes();
	void _upload_mesh(Mesh& mesh);

};
