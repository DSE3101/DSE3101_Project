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

dropdown <- selectInput("dropdownMenu", 
                        "Choose an Option:", 
                        choices = c("Covid-19 Recession" = "opt1", 
                                    "Lehmann Brother's Crash" = "opt2", 
                                    "Great Financial Crisis" = "opt3")
)
Panels
MLPanel <- mainPanel(
  tags$h1("Recurrent Neural Network Model"),
  tags$h2("Description of model"),
  p("A recurrent neural network (RNN) model ")
)

ARPanel <- mainPanel(
  tags$h1("Insert AR Model here"),
  p("Nice")
)

ADLPanel <- mainPanel(
  tags$h1("Insert ADL Model here"),
  p("Nice")
)



#UI
ui <- navbarPage(
  "Time Series Analysis",
  theme = shinytheme("slate"),
  tabPanel("Time Series Graph",
           mainPanel(
             tags$h1("Test header1"),
             tags$h2("Test header2"),
             p("Test Text"),
             div(class = "graphBorder", plotOutput("timeSeriesGraph")),
             slider,
             dropdown)
  ),
  
  tabPanel("Machine Learning Model",
           mainPanel(
             strong("Test!"),
             p("Content for Real Time vs Vintage Data.")
           )
  ),
  
  tabPanel("Time Series Model",
           mainPanel(
             p("Content for another tab.")
           )
  ),
  navbarMenu("Models", 
             tabPanel("Machine Learning", MLPanel),
             tabPanel("ADL", ADLPanel),
             tabPanel("AR", ARPanel)
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
