console.log('script.js loaded');  // Debug
document.getElementById('symptom-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    console.log('Form submitted');  // Debug
    const form = e.target;
    // Force Select2 to update <select> values
    const select = form.querySelector('#symptoms');
    select.dispatchEvent(new Event('change'));
    const formData = new FormData(form);
    // Log all form data entries
    const formDataEntries = Object.fromEntries(formData.entries());
    console.log('Form Data Entries:', formDataEntries);  // Debug
    const symptoms = formData.getAll('symptoms[]');
    console.log('Form Data Symptoms:', symptoms);  // Debug
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        console.log('Response status:', response.status);  // Debug
        const result = await response.json();
        console.log('Response JSON:', result);  // Debug
        const resultDiv = document.getElementById('result');
        if (result.error) {
            console.log('Server error:', result.error);  // Debug
            resultDiv.innerHTML = `<p class="text-red-500 text-center mt-4">${result.error}</p>`;
        } else {
            let html = '<p class="text-green-600 font-bold mb-2 mt-4">Possible Diseases:</p><ul class="list-disc pl-5">';
            result.predictions.forEach(pred => {
                html += `<li>${pred.disease}: ${pred.probability}</li>`;
            });
            html += '</ul>';
            resultDiv.innerHTML = html;
        }
    } catch (error) {
        console.error('JavaScript Error:', error);  // Debug
        document.getElementById('result').innerHTML = '<p class="text-red-500 text-center mt-4">Failed to fetch prediction. Please check the server and try again.</p>';
    }
});