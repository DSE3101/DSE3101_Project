#install.packages('rsconnect')
#install.packages('shinythemes')
library(shiny)
library(tidyverse)
library(ggthemes)
library(lubridate)
library(zoo)
library(scrollytell)
library(rsconnect)
library(shinythemes)


slider = sliderInput("dateRange",
                     "Select Training Date:",
                     min = as.Date("1960-01-01"),
                     max = as.Date("2023-12-31"),
                     value = c(as.Date("1980-01-01"), as.Date("2020-01-01")),
                     timeFormat = "%Y-%m-%d")

checkbox = checkboxGroupInput("checkboxMenu", 
                   "Choose a model variable:", 
                   choices = c("M1" = "opt2", 
                               "Investment" = "opt3",
                               "Total Reserves" = "opt4",
                               "Nonborrowed Reserves" = "opt5",
                               "Nonborrowed Reserves + Extended Credit" = "opt6",
                               "Monetary Base" = "opt7",
                               "Civilian Unemployed Rate" = "opt8",
                               "CPI vs Chain-weighted Price Index" = "opt9",
                               "3-month T Bill Rate" = "opt10",
                               "10-year T-bond Rate" = "opt11")
                   )
#Panels
MLPanel <- mainPanel(
  tags$h1("Recurrent Neural Network Model"),
  tags$h2("Description of model"),
  p("A recurrent neural network (RNN) model ")
)

AboutUsPanel <- mainPanel(
  tags$h1("Insert about us here"),
  p("Nice") ##PLot y-axis = today data, x-axis = real-time
)

GoalPanel <- mainPanel(
  tags$h1("Insert goal here"),
  p("Nice")
)

EvaluationPanel <- mainPanel(
  tags$h1("Insert evaluation here"),
  p("Nice")
)


#UI
ui <- navbarPage(
  "Time Series Analysis",
  theme = shinytheme("slate"),
  tabPanel("Model Training",
           mainPanel(
             tags$h1("Benchmarking Time Series Graph using models"),
             tags$h2("Make your own time series graph"),
             p("In an attempt to make this project more interactive, we are going to allow users to select the
               training data's date and variables they wish to use"),
             div(class = "graphBorder", plotOutput("timeSeriesGraph")),
             slider,
             checkbox,
             actionButton("trainModel", "Train the model!")
           )
  ),
  
  tabPanel("ARIMA",
           mainPanel(
             strong("Test!"),
             p("Content for Real Time vs Vintage Data.")
           )
  ),
  
  tabPanel("QBVC",
           mainPanel(
             p("Content for another tab.")
           )
  ),
  
  tabPanel("ML",
           mainPanel(
             p("Content for another tab.")
           )
  ),
  
  tabPanel("AR",
           mainPanel(
             p("Content for another tab.")
           )
  ),
  navbarMenu("About this project", 
             tabPanel("About us", AboutUsPanel),
             tabPanel("Goal", GoalPanel),
             tabPanel("Evaluation", EvaluationPanel),
  )
)

#Dummy Data
set.seed(123)
dummyData <- data.frame(
  date = seq(as.Date("1960-01-01"), as.Date("2023-12-31"), by = "quarter"),
  value = cumsum(runif(256, min = -10, max = 10)) # Random walk data
)

server <- function(input, output) {
  output$timeSeriesGraph <- renderPlot({
    # Ensure highlightedData uses the filter based on selected date range
    highlightedData <- dummyData %>%
      filter(date >= input$dateRange[1], date <= input$dateRange[2])
    
    # Start plotting using ggplot2 with dummyData as the base data
    plot <- ggplot() +
      geom_line(data = dummyData, aes(x = date, y = value)) + 
      geom_line(data = highlightedData, aes(x = date, y = value), color = "red", size = 1.5) + 
      #Colour the user selected training data red
      labs(title = "Dummy Time Series Graph", x = "Date", y = "Value") +
      theme_economist_white()
    plot
  })
}

shinyApp(ui,server)
