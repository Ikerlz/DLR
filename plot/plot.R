rm(list = ls())
setwd("/home/lizhe/RStudio/FinalPJ")
library(ggplot2)
library(latex2exp)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

scenario1 <- FALSE

if (scenario1 == TRUE) {
  hat <- read.csv("hat_estimator_mat.csv", header = FALSE)
  admm <- read.csv("admm_estimator_mat.csv", header = FALSE)
  pgd <- read.csv("pgd_estimator_mat.csv", header = FALSE)
  sgd <- read.csv("sgd_estimator_mat.csv", header = FALSE)
  sgd[14, ] <- rep(0, 8)
  sgd[15, ] <- rep(0, 8)
  sgd[16, ] <- rep(0, 8)
  sgd[17, ] <- rep(0, 8)
  
  ratio <- rep(c(0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 1, 2, 4, 10, 100, 1000, 5000, 10000), 8)
  group <- c(rep("1", 17), rep("2", 17), rep("3", 17), rep("4", 17),
             rep("5", 17), rep("6", 17), rep("7", 17), rep("8", 17))
  hat_vec <- c(hat[,1], hat[,2], hat[,3], hat[,4],
               hat[,5], hat[,6], hat[,7], hat[,8])
  admm_vec <- c(admm[,1], admm[,2], admm[,3], admm[,4],
                admm[,5], admm[,6], admm[,7], admm[,8])
  pgd_vec <- c(pgd[,1], pgd[,2], pgd[,3], pgd[,4],
               pgd[,5], pgd[,6], pgd[,7], pgd[,8])
  sgd_vec <- c(sgd[,1], sgd[,2], sgd[,3], sgd[,4],
               sgd[,5], sgd[,6], sgd[,7], sgd[,8])
  
  
  plot_hat <- data.frame(ratio=ratio, hat_vec=hat_vec, group=group)
  plot_admm <- data.frame(ratio=ratio, admm_vec=admm_vec, group=group)
  plot_pgd <- data.frame(ratio=ratio, pgd_vec=pgd_vec, group=group)
  plot_sgd <- data.frame(ratio=ratio, sgd_vec=sgd_vec, group=group)
  
  
  
  p1 <- ggplot(data = plot_hat, 
               mapping = aes(x = ratio, y = hat_vec, group = group, colour = group, shape =group)) + 
    geom_line(size=1) +
    geom_point(size=3, stroke=1) + 
    scale_x_log10()+
    scale_color_manual(values = c('#009289', '#FF5030', '#3FA9F5', '#FED254', '#C9C9BF', '#9C661F', '#5E2612', '#082E54'))+
    scale_shape_manual(values = c(0, 1, 2, 3, 4, 5, 6, 7)) +
    #                    labels = c('param1', 'param2', 'param3', 'param4', 'param5'))+
    theme_bw() + 
    labs(x = TeX("$\\lambda$"), y = TeX("$\\beta$")) +
    theme(legend.title=element_text(hjust = 0.5, colour="black", size=15, face="bold"))+
    theme(axis.text.x = element_text(size = 13, face = "bold", vjust = 0.5, hjust = 0.5))+
    theme(axis.text.y = element_text(size = 13, face = 'bold', vjust = 0.5, hjust = 0.5))+
    theme(axis.ticks = element_blank())+
    theme(axis.text=element_text(size=10),
          axis.title.x=element_text(size=18,face="bold"),
          axis.title.y=element_text(size=18,face="bold"),
          legend.text=element_text(size=13),
          plot.title = element_text(size=13)) +
    theme(panel.background = element_rect(colour = "black", size = 1)) +
    theme(legend.position = "none") + 
    theme(legend.key.height = unit(0.8,'cm')) + 
    theme(legend.key.width = unit(1, 'cm'))
  
  
  p2 <- ggplot(data = plot_pgd, 
               mapping = aes(x = ratio, y = pgd_vec, group = group, colour = group, shape =group)) + 
    geom_line(size=1) +
    geom_point(size=3, stroke=1) + 
    scale_x_log10()+
    scale_color_manual(values = c('#009289', '#FF5030', '#3FA9F5', '#FED254', '#C9C9BF', '#9C661F', '#5E2612', '#082E54'))+
    scale_shape_manual(values = c(0, 1, 2, 3, 4, 5, 6, 7)) + 
    #                    labels = c('param1', 'param2', 'param3', 'param4', 'param5'))+
    theme_bw() + 
    labs(x = TeX("$\\lambda$"), y = TeX("$\\beta$")) +
    theme(legend.title=element_text(hjust = 0.5, colour="black", size=15, face="bold"))+
    theme(axis.text.x = element_text(size = 13, face = "bold", vjust = 0.5, hjust = 0.5))+
    theme(axis.text.y = element_text(size = 13, face = 'bold', vjust = 0.5, hjust = 0.5))+
    theme(axis.ticks = element_blank())+
    theme(axis.text=element_text(size=10),
          axis.title.x=element_text(size=18,face="bold"),
          axis.title.y=element_text(size=18,face="bold"),
          legend.text=element_text(size=13),
          plot.title = element_text(size=13)) +
    theme(panel.background = element_rect(colour = "black", size = 1)) +
    theme(legend.position = "none") + 
    theme(legend.key.height = unit(0.8,'cm')) + 
    theme(legend.key.width = unit(1, 'cm'))
  
  
  p3 <- ggplot(data = plot_admm, 
               mapping = aes(x = ratio, y = admm_vec, group = group, colour = group, shape =group)) + 
    geom_line(size=1) +
    geom_point(size=3, stroke=1) + 
    scale_x_log10()+
    scale_color_manual(values = c('#009289', '#FF5030', '#3FA9F5', '#FED254', '#C9C9BF', '#9C661F', '#5E2612', '#082E54'))+
    scale_shape_manual(values = c(0, 1, 2, 3, 4, 5, 6, 7)) + 
    #                    labels = c('param1', 'param2', 'param3', 'param4', 'param5'))+
    theme_bw() + 
    labs(x = TeX("$\\lambda$"), y = TeX("$\\beta$")) +
    theme(legend.title=element_text(hjust = 0.5, colour="black", size=15, face="bold"))+
    theme(axis.text.x = element_text(size = 13, face = "bold", vjust = 0.5, hjust = 0.5))+
    theme(axis.text.y = element_text(size = 13, face = 'bold', vjust = 0.5, hjust = 0.5))+
    theme(axis.ticks = element_blank())+
    theme(axis.text=element_text(size=10),
          axis.title.x=element_text(size=18,face="bold"),
          axis.title.y=element_text(size=18,face="bold"),
          legend.text=element_text(size=13),
          plot.title = element_text(size=13)) +
    theme(panel.background = element_rect(colour = "black", size = 1)) +
    theme(legend.position = "none") + 
    theme(legend.key.height = unit(0.8,'cm')) + 
    theme(legend.key.width = unit(1, 'cm'))
  
  
  p4 <- ggplot(data = plot_sgd, 
               mapping = aes(x = ratio, y = sgd_vec, group = group, colour = group, shape =group)) + 
    geom_line(size=1) +
    geom_point(size=3, stroke=1) + 
    scale_x_log10()+
    scale_color_manual(values = c('#009289', '#FF5030', '#3FA9F5', '#FED254', '#C9C9BF', '#9C661F', '#5E2612', '#082E54'))+
    scale_shape_manual(values = c(0, 1, 2, 3, 4, 5, 6, 7)) +
    #                    labels = c('param1', 'param2', 'param3', 'param4', 'param5'))+
    theme_bw() + 
    labs(x = TeX("$\\lambda$"), y = TeX("$\\beta$")) +
    theme(legend.title=element_text(hjust = 0.5, colour="black", size=15, face="bold"))+
    theme(axis.text.x = element_text(size = 13, face = "bold", vjust = 0.5, hjust = 0.5))+
    theme(axis.text.y = element_text(size = 13, face = 'bold', vjust = 0.5, hjust = 0.5))+
    theme(axis.ticks = element_blank())+
    theme(axis.text=element_text(size=10),
          axis.title.x=element_text(size=18,face="bold"),
          axis.title.y=element_text(size=18,face="bold"),
          legend.text=element_text(size=13),
          plot.title = element_text(size=13)) +
    theme(panel.background = element_rect(colour = "black", size = 1)) +
    theme(legend.position = "none") + 
    theme(legend.key.height = unit(0.8,'cm')) + 
    theme(legend.key.width = unit(1, 'cm'))
  
  img = "xiaohui.png"
  p5 <- ggbackground(p1, img, alpha=0.1)
  p6 <- ggbackground(p2, img, alpha=0.1)
  p7 <- ggbackground(p3, img, alpha=0.1)
  p8 <- ggbackground(p4, img, alpha=0.1)
  
  multiplot(p5, p6, p7, p8, cols = 2)
}


##########################################
## Scenario 2

scenario2 <- FALSE

if (scenario2 == TRUE) {
  data <- read.csv("res_scenario2.csv", header = FALSE)
  data_num <- rep(500 * c(1:20), 5)
  group <- c(rep("1",20),rep("2",20),rep("3",20),rep("4",20),rep("5",20))
  rmse <- c(data[,1], data[,2], data[,3], data[,4], data[,5])
  plot_df <- data.frame(data_num=data_num, rmse=rmse, group=group)
  p1 <- ggplot(data = plot_df, 
               mapping = aes(x = data_num, y = rmse, group = group, colour = group, shape =group)) + 
    geom_line(size=1) +
    geom_point(size=3, stroke=1) + 
    scale_color_manual(values = c('#009289', '#FF5030', '#3FA9F5', '#FED254', '#C9C9BF'))+
    scale_shape_manual(values = c(0, 1, 2, 3, 4)) + 
    #                    labels = c('param1', 'param2', 'param3', 'param4', 'param5'))+
    theme_bw() + 
    labs(x = "Sample Size", y = "RMSE") +
    theme(legend.title=element_text(hjust = 0.5, colour="black", size=15, face="bold"))+
    theme(axis.text.x = element_text(size = 13, face = "bold", vjust = 0.5, hjust = 0.5))+
    theme(axis.text.y = element_text(size = 13, face = 'bold', vjust = 0.5, hjust = 0.5))+
    theme(axis.ticks = element_blank())+
    theme(axis.text=element_text(size=10),
          axis.title.x=element_text(size=18,face="bold"),
          axis.title.y=element_text(size=18,face="bold"),
          legend.text=element_text(size=13),
          plot.title = element_text(size=13)) +
    theme(panel.background = element_rect(colour = "black", size = 1)) +
    theme(legend.position = "none") + 
    theme(legend.key.height = unit(0.8,'cm')) + 
    theme(legend.key.width = unit(1, 'cm'))
  img = "xiaohui.png"
  ggbackground(p1, img, alpha=0.1)
}



##########################################
## Scenario 3

scenario3 <- TRUE

if (scenario3 == TRUE) {
  data <- read.csv("res_scenario3.csv", header = FALSE)
  worker_num <- rep(5 * c(1:20), 5)
  group <- c(rep("1",20),rep("2",20),rep("3",20),rep("4",20),rep("5",20))
  rmse <- c(data[,1], data[,2], data[,3], data[,4], data[,5])
  plot_df <- data.frame(worker_num=worker_num, rmse=rmse, group=group)
  p1 <- ggplot(data = plot_df, 
               mapping = aes(x = worker_num, y = rmse, group = group, colour = group, shape =group)) + 
    geom_line(size=1) +
    geom_point(size=3, stroke=1) + 
    scale_color_manual(values = c('#009289', '#FF5030', '#3FA9F5', '#FED254', '#C9C9BF'))+
    scale_shape_manual(values = c(0, 1, 2, 3, 4)) + 
    #                    labels = c('param1', 'param2', 'param3', 'param4', 'param5'))+
    theme_bw() + 
    labs(x = "Worker Number", y = "RMSE") +
    theme(legend.title=element_text(hjust = 0.5, colour="black", size=15, face="bold"))+
    theme(axis.text.x = element_text(size = 13, face = "bold", vjust = 0.5, hjust = 0.5))+
    theme(axis.text.y = element_text(size = 13, face = 'bold', vjust = 0.5, hjust = 0.5))+
    theme(axis.ticks = element_blank())+
    theme(axis.text=element_text(size=10),
          axis.title.x=element_text(size=18,face="bold"),
          axis.title.y=element_text(size=18,face="bold"),
          legend.text=element_text(size=13),
          plot.title = element_text(size=13)) +
    theme(panel.background = element_rect(colour = "black", size = 1)) +
    theme(legend.position = "none") + 
    theme(legend.key.height = unit(0.8,'cm')) + 
    theme(legend.key.width = unit(1, 'cm'))
  img = "xiaohui.png"
  ggbackground(p1, img, alpha=0.1)
}
