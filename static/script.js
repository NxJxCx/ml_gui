$(function() {
    $("#train-algorithm-select").on("change", function() {
        const url = new URL("/train", window.location.origin);
        url.searchParams.set("algorithm", $(this).val());
        window.location.href = url.toString();  // redirect to specific algorithm to train
    });
    const $loadingOverlay = (/*html*/`
        <div class="spinner-border text-primary loading-overlay" role="status">
            <span class="visually-hidden">Loading...</span>
        </div>
    `);

    var saved_features = []

    function disablePredictSubmitButton() {
        if (!$("#predict-submit-button").prop("disabled")) {
            $("#predict-submit-button").prop("disabled", true);
        }
    }

    function enablePredictSubmitButton() {
        if ($("#predict-submit-button").prop("disabled")) {
            $("#predict-submit-button").prop("disabled", false);
        }
    }


    function showAccordionResults() {
        if (!$($("#results-accordion-button").attr("data-bs-target")).hasClass("show")) {
            $("#results-accordion-button").trigger("click");
        }
    }

    function showAccordionPredict() {
        if (!$($("#predict-accordion-button").attr("data-bs-target")).hasClass("show")) {
            $("#predict-accordion-button").trigger("click");
        }
    }

    function showLoading(id) {
        $(id).empty().append($($loadingOverlay))
    }
    function removeLoading(id) {
        $(id).find(".loading-overlay").remove()
    }
    function addFeaturesInputForPrediction() {
        removeLoading("#predict-input-container");
        $("#predict-input-container").empty()
        saved_features.forEach((feature) => {
            $("#predict-input-container").append(/*html*/`
                <div class="row">
                    <div class="col-md-4 mb-3">
                        <label for="${feature.trim().replaceAll(' ', '')}-feature" class="form-label">${feature}</label>
                        <input type="text" class="form-control" id="${feature.trim().replaceAll(' ', '')}" name="${feature}" placeholder="Enter value for ${feature}" required>
                    </div>
                </div>
            `)
        });
        enablePredictSubmitButton();
    }
    function displayTrainingResults(data) {
        showAccordionResults();
        showAccordionPredict();
        disablePredictSubmitButton();
        const $algorithmElem = $(/*html*/`
            <h4>Algorithm: ${data.algorithm}</h4>
        `);
        const $features = $(/*html*/`
            <div><h6>Features:</h6> <p>[${data.features.join(", ")}]</p></div>
        `);
        const $targets = $(/*html*/`
            <div><h6>Target/s:</h6> <p>[${data.target.join(", ")}]</p></div>
        `);
        const hyperp = Object.entries(data.hyperparameters).filter(([k,v]) => !!v);

        const $hyperparameters = $(/*html*/`
            <div>
                <h6>Hyperparameters used:</h6>
                <div class="my-2 d-flex justify-content-between column-gap-2 row-gap-2 flex-wrap">
                    ${hyperp.map(([k, v]) => {
                        return "<p>" + k + " = " + v + "</p>"
                    }).join("")}
                </div>
            </div>
        `);
        const evals = Object.entries(data.evaluation).filter(([k,v]) => v !== null);
        const $evaluation = $(/*html*/`
            <div class="max-w-100 text-wrap">
                <h6>Evaluations:</h6>
                <div class="my-2 d-flex flex-column justify-content-between column-gap-2 row-gap-2 text-wrap">
                    ${evals.map(([k, v]) => {
                        let val = ""
                        console.log(k, "type:", typeof(v))
                        console.log(v)
                        if (Array.isArray(v)) {
                            val = "<code>";
                            const elem =  v.map((arr) => {
                                if (Array.isArray(arr)) {
                                    return arr.map((arr2) => {
                                        console.log("ARR2:", arr2)
                                        if (Array.isArray(arr2)) {
                                            return arr2.map(item1 => (Math.round(item1 * 1000) / 1000)).join(",").padStart(4, " ").padEnd(6, " ");
                                        } else {
                                            return arr2.toString().padStart(4, " ").padEnd(6, " ");
                                        }
                                    }).join("  |  ")
                                } else if (typeof(arr) === "object") {
                                    return Object.entries(arr).map(([key,val]) => key + " = " + val.toString() + ",").join("    ");
                                }
                                return arr
                            }).join("\n").replaceAll("\n", "<br>").replaceAll(" ", "&nbsp;")
                            val += elem + "</code>"
                        } else {
                            const elem = v.toString().replaceAll("\n", "<br>").replaceAll(" ", "&nbsp;")
                            val = "<code>" + elem + "</code>"
                        }
                        console.log("val is")
                        console.log(val)
                        result = "<h5>" + k.split('_').join(' ').toUpperCase() + "</h5><p class='text-wrap'>" +
                            val.trim() + "</p>"
                        console.log("result now is")
                        console.log(result)
                        return result;
                    }).join("")}
                </div>
            </div>
        `)
        const plotImgs = Object.entries(data.plots).filter(([k, v]) => !!v);
        const $plots = $(/*html*/`
            <div class="max-w-100">
                <h5 class="mx-auto">Plots:</h5>
                <div class="d-flex column-gap-4 row-gap-4 flex-wrap w-100">
                    ${plotImgs.map(([k, v]) => {
                        return  '<div class="d-flex flex-column justify-content-center align-items-start" style="max-width: 600px;">' +
                                    '<h6 class="text-center">' + k.split('_').join(' ').toUpperCase() + '</h6>' +
                                    '<img class="object-fit-contain w-100" src="data:image/png;base64,' + v + '" />' +
                                '</div>'
                    }).join("")}
                </div>
            </div>
        `)

        $("#training-results")
            .append($algorithmElem)
            .append($features)
            .append($targets)
            .append($hyperparameters)
            .append($evaluation)
            .append($plots)

        saved_features = [...data.features]
        addFeaturesInputForPrediction()
        removeLoading("#training-results")
    }

    $("form#training-form").on("submit", function(event) {
        event.preventDefault();
        showLoading("#training-results");
        showLoading("#predict-input-container");
        disablePredictSubmitButton();
        showAccordionResults();
        showAccordionPredict();
        const formData = new FormData();
        const fileInput = $(this).find("input[type=file][name=training_data]");

        if (fileInput[0].files.length === 0) {
            alert('Please select a csv file for the dataset.');
            return;
        }
        const file = fileInput[0].files[0];
        formData.set('training_data', file, file.name);
        const features = $(this).find("input[name=features]").val().split(",").map(v => v?.trim()).filter(v => !!v);
        const targets = $(this).find("input[name=targets]").val().split(",").map(v => v?.trim()).filter(v => !!v);
        formData.append('features', JSON.stringify(features))
        formData.append('target', JSON.stringify(targets))
        const algo = $(this).find("select[name=algorithm]").val();
        formData.set("algo", algo)
        let hyperparameters = Object.fromEntries(new FormData($(this).get(0)));
        delete hyperparameters["training_data"]
        delete hyperparameters["algorithm"]
        delete hyperparameters["features"]
        delete hyperparameters["targets"]
        hyperparameters = Object.fromEntries(Object.entries(hyperparameters).map(([key, value]) => value === "" || (value !== 0 && value !== "0" && !value) ? [key, null] : (
            !Number.isNaN(Number.parseFloat(value)) ? [key, Number.parseFloat(value)] : (
                value === "true" || value === "false" ? [key, value === "true"] : [key, value]
            )
        )))
        formData.set("hyperparameters", JSON.stringify(hyperparameters))

        const xhr = new XMLHttpRequest();
        const url = new URL("/api/train", window.location.origin)
        xhr.open('POST', url.toString(), true);

        xhr.onload = function() {
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.error) {
                        console.log('Error:', response.error);
                        $("#error-message").empty().text(response.error)
                    } else {
                        $("#error-message").empty();
                        displayTrainingResults(response);
                    }
                } catch (e) {
                    console.log('Error parsing JSON:', e);
                    $("#error-message").empty().text(e.toString())
                }
            } else {
                console.log('Error uploading file: ' + xhr.statusText);
                $("#error-message").empty().text(xhr.statusText);
            }
            removeLoading("#training-results");
            removeLoading("#predict-input-container");
        };
        xhr.send(formData);
    })


    $("form#prediction-form").on("submit", function(event) {
        event.preventDefault()
        const formData = new FormData($(this).get(0))
        const bodyData = {
            input: Object.fromEntries(formData),
        }
        bodyData["input"] = Object.fromEntries(Object.entries(bodyData['input']).map(([k,v]) => [k, !Number.isNaN(Number.parseFloat(v)) ? Number.parseFloat(v) : (
            !Number.isNaN(Number.parseInt(v)) ? Number.parseInt(v) : v
        ) ]));
        const url = new URL("/api/predict", window.location.origin)
        $.ajax({
            url: url.toString(),
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(bodyData),
            success: function(response) {
                if (response.error) {
                    console.log("ERROR:", response.error);
                } else {
                    $("#prediction-result").empty().text(response.result.toString());
                }
            },
            error: function(_, statusText) {
                console.log("Error:", statusText);
            }
        });
    })

    function fetchTrainingHistory() {
        if (window.location.pathname === "/train") {
            const url = new URL("/api/train_history", window.location.origin)
            showLoading("#training-results");
            showLoading("#predict-input-container");
            $.get(url.toString())
                .done(function(data) {
                    removeLoading("#training-results");
                    if (data.error) {
                        console.log(data.error)
                    } else{
                        displayTrainingResults(data)
                    }
                })
                .fail(function(_, statusText) {
                    removeLoading("#training-results");
                    console.log("ERROR FETCH:", statusText)
                })
        }
    }
    fetchTrainingHistory()
})

// <div class="col-md-4 mb-3">
// <label for="feature1" class="form-label">Feature 1</label>
// <input type="text" class="form-control" id="feature1" name="feature1" placeholder="Enter value for Feature 1" required>
// <span> : </span>
// <input type="text" class="form-control" id="feature1" name="feature1" placeholder="Enter value for Feature 1" required>
// </div>
