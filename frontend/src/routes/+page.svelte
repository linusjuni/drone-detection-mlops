<script lang="ts">
	type ClassificationResult = 'bird' | 'drone' | null;

	let selectedFile = $state<File | null>(null);
	let imagePreview = $state<string | null>(null);
	let isAnalyzing = $state(false);
	let result = $state<ClassificationResult>(null);
	let isDragOver = $state(false);

	function handleFileSelect(file: File | null) {
		if (!file || !file.type.startsWith('image/')) {
			return;
		}

		selectedFile = file;
		result = null;

		const reader = new FileReader();
		reader.onload = (e) => {
			imagePreview = e.target?.result as string;
		};
		reader.readAsDataURL(file);
	}

	function handleInputChange(event: Event) {
		const input = event.target as HTMLInputElement;
		const file = input.files?.[0] ?? null;
		handleFileSelect(file);
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
		const file = event.dataTransfer?.files[0] ?? null;
		handleFileSelect(file);
	}

	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		isDragOver = true;
	}

	function handleDragLeave() {
		isDragOver = false;
	}

	async function analyzeImage() {
		if (!selectedFile) return;

		isAnalyzing = true;
		result = null;

		console.log('Analyzing image:', selectedFile.name);
		console.log('File size:', selectedFile.size, 'bytes');
		console.log('File type:', selectedFile.type);

		// Simulate API call
		await new Promise((resolve) => setTimeout(resolve, 2000));

		// Mock result (randomly pick bird or drone)
		const mockResult: ClassificationResult = Math.random() > 0.5 ? 'bird' : 'drone';
		console.log('Classification result:', mockResult);

		result = mockResult;
		isAnalyzing = false;
	}

	function reset() {
		selectedFile = null;
		imagePreview = null;
		result = null;
		isAnalyzing = false;
	}
</script>

<div class="flex min-h-screen flex-col bg-redbull-navy">
	<header class="border-b border-redbull-navy-light px-6 py-4">
		<h1 class="text-2xl font-bold tracking-tight text-white">
			<span class="text-redbull-red">Bird</span> or <span class="text-redbull-gold">Drone</span>?
		</h1>
	</header>

	<main class="flex flex-1 flex-col items-center justify-center p-6">
		<div class="w-full max-w-lg">
			{#if !imagePreview}
				<label
					class="flex h-64 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all duration-200 {isDragOver
						? 'border-redbull-gold bg-redbull-navy-light'
						: 'border-redbull-navy-light hover:border-redbull-red hover:bg-redbull-navy-light/50'}"
					ondrop={handleDrop}
					ondragover={handleDragOver}
					ondragleave={handleDragLeave}
				>
					<svg
						class="mb-4 h-12 w-12 text-redbull-silver"
						fill="none"
						stroke="currentColor"
						viewBox="0 0 24 24"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="1.5"
							d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
						/>
					</svg>
					<p class="mb-1 text-sm text-redbull-silver">
						<span class="font-medium text-redbull-red">Click to upload</span> or drag and drop
					</p>
					<p class="text-xs text-redbull-silver/60">PNG, JPG, WEBP</p>
					<input type="file" accept="image/*" class="hidden" onchange={handleInputChange} />
				</label>
			{:else}
				<div class="overflow-hidden rounded-xl bg-redbull-navy-light">
					<div class="relative aspect-video">
						<img
							src={imagePreview}
							alt="Selected"
							class="h-full w-full object-contain"
						/>
						{#if isAnalyzing}
							<div
								class="absolute inset-0 flex items-center justify-center bg-redbull-navy/80"
							>
								<div class="flex flex-col items-center">
									<div
										class="h-10 w-10 animate-spin rounded-full border-4 border-redbull-red border-t-transparent"
									></div>
									<p class="mt-3 text-sm text-white">Analyzing...</p>
								</div>
							</div>
						{/if}
					</div>

					{#if result}
						<div
							class="border-t border-redbull-navy p-6 text-center {result === 'bird'
								? 'bg-redbull-gold/10'
								: 'bg-redbull-red/10'}"
						>
							<p class="text-lg text-redbull-silver">It's a</p>
							<p
								class="text-4xl font-bold uppercase tracking-wider {result === 'bird'
									? 'text-redbull-gold'
									: 'text-redbull-red'}"
							>
								{result}
							</p>
						</div>
					{/if}

					<div class="flex gap-3 border-t border-redbull-navy p-4">
						<button
							onclick={reset}
							class="flex-1 rounded-lg border border-redbull-navy-light px-4 py-2.5 text-sm font-medium text-redbull-silver transition-colors hover:bg-redbull-navy-light"
						>
							Upload New
						</button>
						<button
							onclick={analyzeImage}
							disabled={isAnalyzing}
							class="flex-1 rounded-lg bg-redbull-red px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-redbull-red-dark disabled:cursor-not-allowed disabled:opacity-50"
						>
							{isAnalyzing ? 'Analyzing...' : 'Analyze'}
						</button>
					</div>
				</div>
			{/if}
		</div>
	</main>
</div>
