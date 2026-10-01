const form = document.getElementById("predictionForm");

const result = document.getElementById("result");
const resultText = document.getElementById("resultText");
const probability = document.getElementById("probability");
const resultIcon = document.getElementById("resultIcon");

const resetButton = document.getElementById("resetButton");
const predictButton = document.getElementById("predictButton");

const buttonText = document.getElementById("buttonText");
const spinner = document.getElementById("spinner");


form.addEventListener("submit", async function(event) {

    event.preventDefault();

    console.log("Predict button clicked");

    buttonText.textContent = "Predicting...";
    spinner.classList.remove("hidden");
    predictButton.disabled = true;

    result.classList.add("hidden");


    try {

        const formData = new FormData(form);

        const data = {};

        formData.forEach(function(value, key) {

            data[key] = Number(value);

        });


        console.log("Sending data:", data);


        const response = await fetch("/api/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });


        console.log("Server response:", response.status);


        const resultData = await response.json();

        console.log("Prediction result:", resultData);


        if (!response.ok) {

            throw new Error(
                resultData.error ||
                "Prediction failed."
            );

        }


        resultText.textContent = resultData.result;


        if (
            resultData.probability !== null &&
            resultData.probability !== undefined
        ) {

            probability.textContent =
                resultData.probability + "%";

        } else {

            probability.textContent =
                "Not available";

        }


        if (Number(resultData.prediction) === 1) {

            resultIcon.textContent = "⚠️";

        } else {

            resultIcon.textContent = "✓";

        }


        result.classList.remove("hidden");


        result.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });


    } catch (error) {

        console.error("Prediction error:", error);

        alert(
            "Prediction Error:\n\n" +
            error.message
        );

    } finally {

        buttonText.textContent = "❤️ Predict";

        spinner.classList.add("hidden");

        predictButton.disabled = false;

    }

});


resetButton.addEventListener(
    "click",
    function() {

        form.reset();

        result.classList.add("hidden");

        probability.textContent = "";

        resultText.textContent = "";

        resultIcon.textContent = "";

    }
);