document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById('surveyForm');
    form.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission
        
        // Collect form data
        const formData = new FormData(form);
  
        // Send form data via AJAX
        fetch('/submit', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response from the server
            // Commenting out the alert message
            // if (data.success) {
            //     alert('Form submitted successfully!');
            // } else {
            //     alert('Form submission failed!');
            // }
        })
        .catch(error => {
            console.error('Error:', error);
            // Display error message or perform any other action
            alert('An error occurred while submitting the form.');
        });
    });
  });
  