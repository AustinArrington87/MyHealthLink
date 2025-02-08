{% extends "base.html" %}

{% block title %}Home{% endblock %}

{% block content %}
<div class="max-w-4xl mx-auto">
    <h1 class="text-3xl font-bold mb-8">Welcome, Michelle G</h1>
    
    <div class="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 class="text-xl font-semibold mb-4">Analyze Health Record</h2>
        <div class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center" id="dropZone">
            <input type="file" id="fileInput" multiple class="hidden" accept=".pdf,.png,.jpg,.jpeg">
            <button onclick="document.getElementById('fileInput').click()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                + Add Document
            </button>
            <p class="text-gray-500 mt-2">or drag and drop files here</p>
        </div>
    </div>

    <div id="analysisResult" class="bg-white shadow-md rounded-lg p-6 mb-6 hidden">
        <h2 class="text-xl font-semibold mb-4">Analysis Results</h2>
        <div class="space-y-6">
            <div>
                <h3 class="font-medium mb-3">Synopsis</h3>
                <p id="synopsis" class="text-gray-700 whitespace-pre-line leading-relaxed"></p>
            </div>
            <div>
                <h3 class="font-medium mb-3">Insights and Anomalies</h3>
                <div class="bg-white rounded-lg">
                    <p id="insights-anomalies" class="text-gray-700 whitespace-pre-line leading-relaxed"></p>
                </div>
            </div>
            <div id="citationsSection" class="hidden">
                <h3 class="font-medium mb-3">Citations</h3>
                <div class="bg-gray-50 border-l-4 border-blue-500 p-4 rounded-r-lg">
                    <div class="flex items-start">
                        <div class="flex-shrink-0 mt-1">
                            <svg class="h-5 w-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                        </div>
                        <div id="citations" class="ml-3 text-gray-700 whitespace-pre-line text-sm leading-relaxed"></div>
                    </div>
                </div>
            </div>
            <div class="flex space-x-4 mt-6">
                <button class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600" onclick="saveAnalysis()">
                    Save
                </button>
                <button class="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600" onclick="shareAnalysis()">
                    Share
                </button>
            </div>
        </div>
    </div>

    <div class="bg-white shadow-md rounded-lg p-6">
        <h2 class="text-xl font-semibold mb-4">My Analysis History</h2>
        <div id="analysisHistory" class="space-y-4">
            <!-- Analysis history items will be populated here -->
        </div>
    </div>
</div>

<div id="loadingState" class="hidden">
    <div class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white p-8 rounded-lg shadow-xl max-w-md w-full mx-4">
            <div class="flex items-center justify-center mb-4">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
            
            <div class="text-center">
                <h3 class="text-lg font-semibold mb-2">AI Analysis in Progress</h3>
                <p id="loadingQuote" class="text-gray-600 transition-opacity duration-500"></p>
                <p id="almostReady" class="text-gray-600 hidden transition-opacity duration-500">
                    Your results are almost ready...
                </p>
                <div class="mt-4 flex justify-center space-x-2">
                    <span class="animate-pulse delay-100 text-blue-500">●</span>
                    <span class="animate-pulse delay-200 text-blue-500">●</span>
                    <span class="animate-pulse delay-300 text-blue-500">●</span>
                </div>
            </div>
        </div>
    </div>
</div>

<script src="/static/js/quotes.js"></script>

<script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-blue-500');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-blue-500');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-blue-500');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files[]', file);
        }

        // Show loading state
        const loadingState = document.getElementById('loadingState');
        const loadingQuote = document.getElementById('loadingQuote');
        const almostReady = document.getElementById('almostReady');
        loadingState.classList.remove('hidden');
        
        // Get a random quote
        const randomQuote = wellnessQuotes[Math.floor(Math.random() * wellnessQuotes.length)];
        loadingQuote.textContent = randomQuote;
        
        // After 4 seconds, transition to "almost ready" message
        setTimeout(() => {
            loadingQuote.classList.add('opacity-0');
            setTimeout(() => {
                loadingQuote.classList.add('hidden');
                almostReady.classList.remove('hidden');
                setTimeout(() => {
                    almostReady.classList.remove('opacity-0');
                }, 50);
            }, 500);
        }, 4000);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingState.classList.add('hidden');
            // Reset the loading state elements for next time
            loadingQuote.classList.remove('hidden', 'opacity-0');
            almostReady.classList.add('hidden', 'opacity-0');
            
            resetUploadArea();
            if (data && data.success) {
                showAnalysisResult(data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        })
        .catch(error => {
            loadingState.classList.add('hidden');
            // Reset the loading state elements for next time
            loadingQuote.classList.remove('hidden', 'opacity-0');
            almostReady.classList.add('hidden', 'opacity-0');
            
            console.error('Error:', error);
            resetUploadArea();
            alert('An error occurred while analyzing the files. Please try again.');
        });
    }

    function showAnalysisResult(data) {
        const resultDiv = document.getElementById('analysisResult');
        resultDiv.classList.remove('hidden');
        
        try {
            if (data && data.result) {
                // Show synopsis
                document.getElementById('synopsis').textContent = 
                    data.result.synopsis || 'No synopsis available';
                
                // Show insights and anomalies
                const insightsAnomalies = data.result.insights_anomalies;
                document.getElementById('insights-anomalies').textContent = 
                    insightsAnomalies || 'No significant findings to report.';
                
                // Handle citations
                const citationsSection = document.getElementById('citationsSection');
                const citations = data.result.citations;
                
                if (citations && citations !== 'No citations available') {
                    document.getElementById('citations').textContent = citations;
                    citationsSection.classList.remove('hidden');
                } else {
                    citationsSection.classList.add('hidden');
                }
            } else {
                throw new Error('Invalid data format');
            }
        } catch (error) {
            console.error('Error processing results:', error);
            document.getElementById('synopsis').textContent = 'Error processing analysis';
            document.getElementById('insights-anomalies').textContent = 'Error processing analysis';
            citationsSection.classList.add('hidden');
        }
    }

    function resetUploadArea() {
        dropZone.innerHTML = `
            <input type="file" id="fileInput" multiple class="hidden" accept=".pdf,.png,.jpg,.jpeg">
            <button onclick="document.getElementById('fileInput').click()" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                + Add Document
            </button>
            <p class="text-gray-500 mt-2">or drag and drop files here</p>
        `;
        
        // Reattach event listener to new file input
        const newFileInput = document.getElementById('fileInput');
        newFileInput.addEventListener('change', (e) => handleFiles(e.target.files));
    }

    function saveAnalysis() {
        alert('Analysis saved successfully!');
    }

    function shareAnalysis() {
        alert('Share feature coming soon!');
    }
</script>

<style>
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .animate-pulse {
        animation: pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    .delay-100 {
        animation-delay: 0.1s;
    }
    
    .delay-200 {
        animation-delay: 0.2s;
    }
    
    .delay-300 {
        animation-delay: 0.3s;
    }
    
    .opacity-0 {
        opacity: 0;
    }
    
    .transition-opacity {
        transition-property: opacity;
        transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .duration-500 {
        transition-duration: 500ms;
    }
</style>
{% endblock %}
