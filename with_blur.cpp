#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <pthread.h>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//mutex для синхронизации потоков
static pthread_mutex_t foo_mutex = PTHREAD_MUTEX_INITIALIZER;

struct thread_data
{
  string path;
  int thread_id;
  string window_title; //уникальный window title для каждого потока
};

struct threadRead_data
{
  string path;
  Mat *frame;
  VideoCapture cap;
};

void *ReadFrame(void *threadarg) // основная работающая функция считывания кадра дополнительным потоком
{
  struct threadRead_data *data;
  data = (struct threadRead_data *)threadarg;

  int empty_frame_counts = 0;
  int opening_counts=0;

  while (true)
  {
    data->cap.set(CAP_PROP_BUFFERSIZE, 1);
    data->cap >> *data->frame;
    
    if (data->frame->empty())
    {
      empty_frame_counts++;
      if (empty_frame_counts > 10)
      {
        data->cap.release();
        while (!data->cap.isOpened() && opening_counts <= 10)
        {
          //безопасное открытие видеопотока
          pthread_mutex_lock(&foo_mutex);
          data->cap.open(data->path);
          pthread_mutex_unlock(&foo_mutex);
          opening_counts++;
        }
        opening_counts = 0;
        if (!data->cap.isOpened())
        {
          //cout << "Не удалось открыть " << data->thread_id << " камеру" << endl;
          data->cap.release();
          pthread_exit(NULL); // если камера не открылась, закрываем поток
        }
      }
      continue;
    }
  }
}

void *capture(void *threadarg)
{
  struct thread_data *data;
  data = (struct thread_data *)threadarg;

  VideoCapture cap;

  int empty_frame_counts = 0, opening_counts = 0;

  //цикл попыткок открытия камеры
  while (!cap.isOpened() && opening_counts <= 10)
  {
    //безопасное открытие видеопотока
    pthread_mutex_lock(&foo_mutex);
    //cap.open(data->path, CAP_FFMPEG);
    cap.open(data->path);
    pthread_mutex_unlock(&foo_mutex);
    opening_counts++;
  }

  if (!cap.isOpened())
  {
    cout << "Не удалось открыть " << data->thread_id << " камеру" << endl;
    cap.release();
    pthread_exit(NULL); // если камера не открылась, закрываем поток
  }
  cout << "IP camera " << data->thread_id << " открыта!" << endl;

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  //вычитание фона
  Ptr<BackgroundSubtractor> pBackSub;
  pBackSub = createBackgroundSubtractorMOG2(100, 16, false);

  Mat frame_local;
  Mat frame;
  Mat fgMask; //маска переднего плана

  //////////////////   ////////////////// - потоки
  struct threadRead_data td;
  pthread_t threads;

  int rc = 0;

  td.frame = &frame;
  td.cap = cap;
  td.path=data->path;
  rc = pthread_create(&(threads), NULL, ReadFrame, (void *)&(td));

  if (rc)
  {
    cout << "Ошибка: невозможно создать поток считывания кадра "<<data->thread_id<<"-ой IP камеры" << rc << endl;
    exit(-1);
  }
  //////////////////   //////////////////

  RNG rng(12345);


  vector<Rect> boundRect;
  vector<vector<Point>> contours_poly;
  
  while (true)
  {
    frame_local = frame;

    if (!frame.empty())
    {
      // конвертируем в оттенки серого и устанавливаем первый кадр

      cvtColor(frame_local, frame_local, COLOR_BGR2GRAY);
      GaussianBlur(frame_local, frame_local, Size(21, 21), 0);

      //обновить фоновую модель
      pBackSub->apply(frame_local, fgMask);

      //внешние контуры, метод простой цепной аппроксимации - 4 точки
      findContours(fgMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

      boundRect.resize(contours.size());
      contours_poly.resize(contours.size());
      for (int i = 0; i < contours.size(); i++)
      {
        if (contourArea(contours[i]) < 500)
        {
          continue;
        }
        cout << "Движение обнаружено на камере " << data->thread_id << endl;

        putText(frame_local, "Motion Detected", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
        //аппроксимация контура
        approxPolyDP(contours[i], contours_poly[i], 3, true);

        //вычисление прямоугольника с минимальной площадью
        boundRect[i] = boundingRect(contours_poly[i]);
        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        rectangle(frame_local, boundRect[i].tl(), boundRect[i].br(), color, 2);

        /* cout<<"координаты "<<boundRect[i].tl()<< " "<<boundRect[i].br()<<endl;
      cout<< "area  "<<boundRect[i].area()<<endl;
      cout<< "x  "<<boundRect[i].x<< " y "<<boundRect[i].y<<endl;
      cout<< "height   "<<boundRect[i].height<<" width " <<boundRect[i].width<<endl;
     */
      }

      // вывод кодека
      /*int ex = static_cast<int>(cap.get(CAP_PROP_FOURCC));
      cout << ex << "Input : " << cap.get(CAP_PROP_FOURCC) << endl;
      char EXT[] = {(char)(ex & 0XFF), (char)((ex & 0XFF00) >> 8), (char)((ex & 0XFF0000) >> 16), (char)((ex & 0XFF000000) >> 24), 0};
      cout << "Input codec type: " << EXT << endl;*/

      //resize(frame_local, frame_local, Size(), 0.5, 0.5);
      imshow(data->window_title, frame_local);

      //resize(fgMask, fgMask, Size(), 0.5, 0.5);
      imshow("mask " + data->window_title, fgMask);
    }
    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27)
      break;

    this_thread::sleep_for(100ms);
  }

  pthread_join(threads, NULL);
  //освобождение объекта VideoCapture
  cap.release();
  //уничтожить ранее созданное окно

  //закрыть поток
  cout << data->thread_id << " поток завершился" << endl;
  pthread_exit(NULL);
}


int divideImage(const cv::Mat &img, int x, int y, std::vector<cv::Mat> &blocks)
{
  // Checking if the image was passed correctly
  if (!img.data || img.empty())
  {
    std::wcout << "Image Error: Cannot load image to divide." << std::endl;
    return EXIT_FAILURE;
  }

  // init image dimensions
  int imgWidth = img.cols;
  int imgHeight = img.rows;
  //std::wcout << "IMAGE SIZE: " << "(" << imgWidth << "," << imgHeight << ")" << std::endl;

  int blockWidth = imgWidth / x;
  int blockHeight = imgHeight / y;

  // init block dimensions
  int bwSize;
  int bhSize;

  bool flagX = false;
  bool flagY = false;

  int y0 = 0;
  while (y0 < imgHeight)
  {
    // compute the block height
    bhSize = ((y0 + blockHeight) > imgHeight) * (blockHeight - (y0 + blockHeight - imgHeight)) + ((y0 + blockHeight) <= imgHeight) * blockHeight;

    int tempY0 = y0 + blockHeight;
    int tempBhSize = ((tempY0 + blockHeight) > imgHeight) * (blockHeight - (tempY0 + blockHeight - imgHeight)) + ((tempY0 + blockHeight) <= imgHeight) * blockHeight;
    if (tempBhSize != bhSize)
    {
      bhSize += tempBhSize;
      flagY = true;
    }

    int x0 = 0;
    while (x0 < imgWidth)
    {
      // compute the block height
      bwSize = ((x0 + blockWidth) > imgWidth) * (blockWidth - (x0 + blockWidth - imgWidth)) + ((x0 + blockWidth) <= imgWidth) * blockWidth;

      //////////////////
      int tempX0 = x0 + blockWidth;
      int tempBwSize = ((tempX0 + blockWidth) > imgWidth) * (blockWidth - (tempX0 + blockWidth - imgWidth)) + ((tempX0 + blockWidth) <= imgWidth) * blockWidth;
      if (tempBwSize != bwSize)
      {
        bwSize += tempBwSize;
        flagX = true;
      }
      /////////////////////

      //if (flag)
        //cout << " x0 " << x0 << " y0 " << y0 << " bwSize " << bwSize << " bhSize " << bhSize << " " << endl;

      // crop block
      blocks.push_back(img(cv::Rect(x0, y0, bwSize, bhSize)).clone());

      // update x-coordinate
      if (!flagX)
        x0 = x0 + blockWidth;
      else
      {
        x0 = (x0 + blockWidth) * 2;
        flagX = false;
      }
    }

    // update y-coordinate
    if (!flagY)
      y0 = y0 + blockHeight;
    else
      y0 = (y0 + blockHeight) * 2;
  }
  //flag = false;
  return EXIT_SUCCESS;
}

void *captureBlocks(void *threadarg) //разделение на
{
  struct thread_data *data;
  data = (struct thread_data *)threadarg;

  VideoCapture cap;

  int opening_counts = 0;

  //цикл попыткок открытия камеры
  while (!cap.isOpened() && opening_counts <= 10)
  {
    //безопасное открытие видеопотока
    pthread_mutex_lock(&foo_mutex);
    //cap.open(data->path, CAP_FFMPEG);
    cap.open(data->path);
    pthread_mutex_unlock(&foo_mutex);
    opening_counts++;
  }

  opening_counts = 0;
  if (!cap.isOpened())
  {
    cout << "Не удалось открыть " << data->thread_id << " камеру" << endl;
    cap.release();
    pthread_exit(NULL); // если камера не открылась, закрываем поток
  }
  cout << "IP camera " << data->thread_id << " открыта!" << endl;

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  //вычитание фона

  //Ptr<BackgroundSubtractorKNN> pBackSub;
  //pBackSub = createBackgroundSubtractorMOG2(50, 16, false); // !!!!!!!!!!! посмотреть на производительность без теней
  //pBackSub =  createBackgroundSubtractorKNN();

  Mat frame_local;
  Mat frame;
  Mat fgMask; //маска переднего плана

  //////////////////   ////////////////// - потоки
  struct threadRead_data td;
  pthread_t threads;

  int rc = 0;

  td.frame = &frame;
  td.cap = cap;
  rc = pthread_create(&(threads), NULL, ReadFrame, (void *)&(td));

  if (rc)
  {
    cout << "Ошибка: невозможно создать поток," << rc << endl;
    exit(-1);
  }
  //////////////////   //////////////////
  cout << "1" << endl;
  RNG rng(12345);

  vector<Rect> boundRect;
  vector<vector<Point>> contours_poly;

  std::vector<cv::Mat> blocks;

  int countBlocksW = 10; //кол-во блоков по X
  int countBlocksH = 10; //кол-во блоков по Y

  int quantSquare = 0;

 // vector<int> index;
  vector<int> index={1,3,4,5 ,6 ,7 ,8 ,9 ,10 ,11 ,12 ,13 ,14, 15};


  vector<Ptr<BackgroundSubtractor>> pBackSubs;
  for (int i = 0; i < countBlocksW * countBlocksH; i++)
  {
   // index.push_back(i);
    pBackSubs.push_back(createBackgroundSubtractorMOG2(50, 16, false));
  }

  while (true)
  {
    frame_local = frame;
    //cout << "frame_local " << frame_local.size << endl;

    if (!frame_local.empty())
    {
      cvtColor(frame_local, frame_local, COLOR_BGR2GRAY);
      if (divideImage(frame_local, countBlocksW, countBlocksH, blocks))
        break;

      for (int j : index)
      {
        // конвертируем в оттенки серого и устанавливаем первый кадр

        //int start = getTickCount();
        //cvtColor(frame, frame, COLOR_BGR2GRAY); //изначальный вариант
        //GaussianBlur(frame_local, frame_local, Size(21, 21), 0);
        //int end = getTickCount();
        //cout << "tik" << end - start << endl;

        //обновить фоновую модель
        //при добавлении размытия blurFrame заменить на frame
        //int start = getTickCount();
        imshow(to_string(j), blocks[j]);
        pBackSubs[j]->apply(blocks[j], blocks[j]);
        //int end = getTickCount();
        //cout << "tik" << end-start << endl;

        //внешние контуры, метод простой цепной аппроксимации - 4 точки
        //int start = getTickCount();
        findContours(blocks[j], contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
        //int end = getTickCount();
        //cout << "tik" << end - start << endl;

        //boundRect.resize(contours.size());
        //contours_poly.resize(contours.size());
        for (int i = 0; i < contours.size(); i++)
        {
          if (contourArea(contours[i]) < 500)
          {
            continue;
          }
          cout << "Движение обнаружено на камере " << data->thread_id << " на экране" << j << endl;

          //int start = getTickCount();
          //putText(block, "Motion Detected", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
          //аппроксимация контура
          //approxPolyDP(contours[i], contours_poly[i], 3, true);

          //вычисление прямоугольника с минимальной площадью
          //boundRect[i] = boundingRect(contours_poly[i]);
          //Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
          //rectangle(block, boundRect[i].tl(), boundRect[i].br(), color, 2);
          //int end = getTickCount();
          //cout << "tik" << end - start << endl;

          //cout << "координаты " << boundRect[i].tl() << " " << boundRect[i].br() << endl;
          //cout << "area  " << boundRect[i].area() << endl;
          //cout << "x  " << boundRect[i].x << " y " << boundRect[i].y << endl;
          //cout << "height   " << boundRect[i].height << " width " << boundRect[i].width << endl;
        }
      }

      // вывод кодека
      /*int ex = static_cast<int>(cap.get(CAP_PROP_FOURCC));
      cout << ex << "Input : " << cap.get(CAP_PROP_FOURCC) << endl;
      char EXT[] = {(char)(ex & 0XFF), (char)((ex & 0XFF00) >> 8), (char)((ex & 0XFF0000) >> 16), (char)((ex & 0XFF000000) >> 24), 0};
      cout << "Input codec type: " << EXT << endl;*/

      //resize(frame_local, frame_local, Size(), 0.5, 0.5);
      //int start = getTickCount();
      //imshow(data->window_title, frame_local);

      //resize(fgMask, fgMask, Size(), 0.5, 0.5);
      //imshow("mask " + data->window_title, fgMask);
      //int end = getTickCount();
      //cout << "tik" << end - start << endl;
    }
    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27)
      break;

    this_thread::sleep_for(100ms);
    blocks.clear();
  }

  pthread_join(threads, NULL);
  //освобождение объекта VideoCapture
  cap.release();

  //закрыть поток
  cout << data->thread_id << " поток завершился" << endl;
  pthread_exit(NULL);
}


int divideImageRaspb(const cv::Mat &img, int x, int y, std::vector<cv::Mat> &blocks)
{
  // Checking if the image was passed correctly
  if (!img.data || img.empty())
  {
    std::wcout << "Image Error: Cannot load image to divide." << std::endl;
    return EXIT_FAILURE;
  }

  // init image dimensions
  int imgWidth = img.cols;
  int imgHeight = img.rows;
  //std::wcout << "IMAGE SIZE: " << "(" << imgWidth << "," << imgHeight << ")" << std::endl;

  int blockWidth = imgWidth / x;
  int blockHeight = imgHeight / y;

  // init block dimensions
  int bwSize;
  int bhSize;

  bool flagX = false;
  bool flagY = false;

  int y0 = 0;
  while (y0 < imgHeight)
  {
    // compute the block height
    bhSize = ((y0 + blockHeight) > imgHeight) * (blockHeight - (y0 + blockHeight - imgHeight)) + ((y0 + blockHeight) <= imgHeight) * blockHeight;

    int tempY0 = y0 + blockHeight;
    int tempBhSize = ((tempY0 + blockHeight) > imgHeight) * (blockHeight - (tempY0 + blockHeight - imgHeight)) + ((tempY0 + blockHeight) <= imgHeight) * blockHeight;
    if (tempBhSize != bhSize)
    {
      bhSize += tempBhSize;
      flagY = true;
    }

    int x0 = 0;
    while (x0 < imgWidth)
    {
      // compute the block height
      bwSize = ((x0 + blockWidth) > imgWidth) * (blockWidth - (x0 + blockWidth - imgWidth)) + ((x0 + blockWidth) <= imgWidth) * blockWidth;

      //////////////////
      int tempX0 = x0 + blockWidth;
      int tempBwSize = ((tempX0 + blockWidth) > imgWidth) * (blockWidth - (tempX0 + blockWidth - imgWidth)) + ((tempX0 + blockWidth) <= imgWidth) * blockWidth;
      if (tempBwSize != bwSize)
      {
        bwSize += tempBwSize;
        flagX = true;
      }
      /////////////////////


      // crop block
      blocks.push_back(img(cv::Rect(x0, y0, bwSize, bhSize)).clone());

      // update x-coordinate
      if (!flagX)
        x0 = x0 + blockWidth;
      else
      {
        x0 = (x0 + blockWidth) * 2;
        flagX = false;
      }
    }

    // update y-coordinate
    if (!flagY)
      y0 = y0 + blockHeight;
    else
      y0 = (y0 + blockHeight) * 2;
  }
  return EXIT_SUCCESS;
}

void *captureBlocksRaspb(void *threadarg) //разделение на
{
  struct thread_data *data;
  data = (struct thread_data *)threadarg;

  VideoCapture cap;

  int opening_counts = 0;

  //цикл попыткок открытия камеры
  while (!cap.isOpened() && opening_counts <= 10)
  {
    //безопасное открытие видеопотока
    pthread_mutex_lock(&foo_mutex);
    //cap.open(data->path, CAP_FFMPEG);
    cap.open(data->path);
    pthread_mutex_unlock(&foo_mutex);
    opening_counts++;
  }

  opening_counts = 0;
  if (!cap.isOpened())
  {
    cout << "Не удалось открыть " << data->thread_id << " камеру" << endl;
    cap.release();
    pthread_exit(NULL); // если камера не открылась, закрываем поток
  }
  cout << "IP camera " << data->thread_id << " открыта!" << endl;

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  Mat frame_local;
  Mat frame;
  Mat fgMask; //маска переднего плана

  //////////////////   ////////////////// - потоки
  struct threadRead_data td;
  pthread_t threads;

  int rc = 0;

  td.frame = &frame;
  td.cap = cap;
  rc = pthread_create(&(threads), NULL, ReadFrame, (void *)&(td));

  if (rc)
  {
    cout << "Ошибка: невозможно создать поток," << rc << endl;
    exit(-1);
  }
  //////////////////   //////////////////
  RNG rng(12345);

  vector<Rect> boundRect;
  vector<vector<Point>> contours_poly;

  std::vector<cv::Mat> blocks;

  int countBlocksW = 10; //кол-во блоков по X
  int countBlocksH = 10; //кол-во блоков по Y

  int quantSquare = 0;

  //vector<int> index;
  vector<int> index = {1, 3, 4};

  vector<Ptr<BackgroundSubtractor>> pBackSubs;
  for (int i = 0; i < countBlocksW * countBlocksH; i++)
  {
    //index.push_back(i);
    pBackSubs.push_back(createBackgroundSubtractorMOG2(50, 16, false));
  }

  while (true)
  {
    frame_local = frame;

    if (!frame_local.empty())
    {
      cvtColor(frame_local, frame_local, COLOR_BGR2GRAY);
      if (divideImageRaspb(frame_local, countBlocksW, countBlocksH, blocks))
        break;



      for (int j : index)
      {
        //imshow(to_string(j), blocks[j]);
        pBackSubs[j]->apply(blocks[j], blocks[j]);

        findContours(blocks[j], contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

        for (int i = 0; i < contours.size(); i++)
        {
          if (contourArea(contours[i]) < 500)
          {
            continue;
          }
          cout << "Движение обнаружено на камере " << data->thread_id << " на экране" << j << endl;
        }
      }
    }
    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27)
      break;

    this_thread::sleep_for(100ms);
    blocks.clear();
  }

  pthread_join(threads, NULL);
  cap.release();

  //закрыть поток
  cout << data->thread_id << " поток завершился" << endl;
  pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
  int thread_count = 0;

  if (argc < 2)
  {
    cout << "Введите имя файла с путями камер" << endl;
    return 0;
  }

  string str_in_file;
  vector<string> camers_name;

  //окрываем файл для чтения
  ifstream in(argv[1]);
  if (in.is_open())
  {
    while (getline(in, str_in_file))
    {
      camers_name.push_back(str_in_file);
    }
  }
  //закрываем файл
  in.close();

  thread_count = camers_name.size();
  pthread_t threads[thread_count];
  struct thread_data td[thread_count];

  int rc = 0;
  for (int i = 0; i < thread_count; i++)
  {
    cout << "Main: поток " << i << " создан" << endl;
    td[i].thread_id = i;
    td[i].path = camers_name[i];
    td[i].window_title = to_string(i);

    rc = pthread_create(&(threads[i]), NULL, captureBlocks, (void *)&(td[i]));

    if (rc)
    {
      cout << "Ошибка: невозможно создать поток," << rc << endl;
      exit(-1);
    }
  }

  //ждем пока ранее созданные потоки завершат выполнение
  for (int i = 0; i < thread_count; i++)
  {
    pthread_join(threads[i], NULL);
  }

  pthread_exit(NULL);

  return 0;
}
