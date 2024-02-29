#FrontEnd Code

#Main Visuals = Time Series Graph
#Components we need to work on
#Use sliders to implement a time period slider to choose a time horizon to train the data
library(shiny)

#Slider component
# Define UI
Main <- fluidPage(
  titlePanel("Time Series"),
  sidebarLayout(
    sidebarPanel(
      # Create a two-sided slider for date range selection
      sliderInput("dateRange",
                  "Select your training data:",
                  min = as.Date("1960-01-01"),
                  max = as.Date("2023-12-31"),
                  value = c(as.Date("1960-01-01"), as.Date("2022-12-31")),
                  timeFormat = "%Y-%m-%d")
    ),
    mainPanel(
      textOutput("selectedPeriod")
    )
  )
)

# Define server logic
server <- function(input, output) {
  output$selectedPeriod <- renderText({
    paste("Training data range", format(input$dateRange[1]), "to", format(input$dateRange[2]))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)

