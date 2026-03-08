using System.Diagnostics;
using OpenCvSharp;
using Sdcb.PaddleOCR;
using Sdcb.PaddleOCR.Models;
using SkiaSharp;

var appPath = AppContext.BaseDirectory;
var detPath = Path.Combine(appPath, "model/det");
var recPath = Path.Combine(appPath, "model/rec");
var clsPath = Path.Combine(appPath, "model/cls");

OneByOne();


/// <summary>
/// 将三个模型拆分开一个一个使用
/// </summary>
void OneByOne()
{
    using var detModel = new PaddleOcrDetector(DetectionModel.FromDirectory(detPath, ModelVersion.V5));
    using var recModel = new PaddleOcrRecognizer(RecognizationModel.FromDirectoryV5(recPath));
    using var clsModel = new PaddleOcrClassifier(ClassificationModel.FromDirectory(clsPath));
    string imagePath = Path.Combine(appPath, "images/demo.png");
    using Mat src = Cv2.ImRead(imagePath);
    // 首先检测文字区域
    var result = detModel.Run(src);
    foreach (var region in result)
    {
        // 获取点计算面积
        Point[] pts = region.Points().Select(p => new Point(p.X, p.Y)).ToArray();
        var area = GetArea(pts);
        if (area < 800)
            continue;
        // 裁剪出文字区域（旋转）
        Mat roi = PaddleOcrAll.GetRotateCropImage(src, region);
        var clsResult = clsModel.Run(roi);
        // 识别文字
        var recResult = recModel.Run(clsResult);
        if (string.IsNullOrEmpty(recResult.Text))
            continue;
        // 画框
        Cv2.Polylines(src, new[] { pts }, isClosed: true, color: Scalar.Red, thickness: 2);
        // 写字
        if (!string.IsNullOrEmpty(recResult.Text))
            DrawChineseTextSafe(src, recResult.Text, pts[0], 20, Scalar.Blue);
    }
    using (new Window("OCR Debug", src))
    {
        Cv2.WaitKey();
    }
}

/// <summary>
/// 使用包中的all一次性识别
/// </summary>
void UseAll()
{
    FullOcrModel model = new FullOcrModel(
        DetectionModel.FromDirectory(detPath, ModelVersion.V5),
        ClassificationModel.FromDirectory(clsPath),
        RecognizationModel.FromDirectoryV5(recPath));
    using (PaddleOcrAll all = new PaddleOcrAll(model)
    {
        Enable180Classification = true,
        AllowRotateDetection = true,
    })
    {
        string imagePath = Path.Combine(appPath, "images/demo.png");
        using (Mat src = Cv2.ImRead(imagePath))
        {
            PaddleOcrResult result = all.Run(src);
            foreach (PaddleOcrResultRegion region in result.Regions)
            {
                // 获取点计算面积
                Point[] pts = region.Rect.Points().Select(p => new Point(p.X, p.Y)).ToArray();
                var area = GetArea(pts);
                if (area < 800 || string.IsNullOrEmpty(region.Text))
                    continue;
                // 画框
                Cv2.Polylines(src, new[] { pts }, isClosed: true, color: Scalar.Red, thickness: 2);
                // 写字
                if (!string.IsNullOrEmpty(region.Text))
                    DrawChineseTextSafe(src, region.Text, pts[0], 20, Scalar.Blue);
            }
            using (new Window("OCR Debug", src))
            {
                Cv2.WaitKey();
            }
        }
    }
}



/// <summary>
/// 计算面积，用于筛选掉太小的轮廓
/// </summary>
/// <param name="points"></param>
double GetArea(Point[] points)
{
    if (points.Length < 4)
        return 0;
    double area = 0;
    int j = points.Length - 1;
    for (int i = 0; i < points.Length; i++)
    {
        area += (points[j].X + points[i].X) * (points[j].Y - points[i].Y);
        j = i;
    }
    return Math.Abs(area / 2.0);
}

void DrawChineseTextSafe(Mat mat, string text, Point pos, int fontSize, Scalar color)
{
    using Mat bgraMat = new Mat();
    bool is3Channel = mat.Channels() == 3;

    if (is3Channel)
    {
        Cv2.CvtColor(mat, bgraMat, ColorConversionCodes.BGR2BGRA);
    }
    else
    {
        mat.CopyTo(bgraMat);
    }
    var info = new SKImageInfo(bgraMat.Width, bgraMat.Height, SKColorType.Bgra8888, SKAlphaType.Premul);

    using (var bitmap = new SKBitmap())
    {
        bitmap.InstallPixels(info, bgraMat.Data);


        using var canvas = new SKCanvas(bitmap);
        using var paint = new SKPaint()
        {
            Color = new SKColor((byte)color.Val2, (byte)color.Val1, (byte)color.Val0),
            IsAntialias = true,
        };
        using var font = new SKFont()
        {
            Size = fontSize,
            Typeface = SKTypeface.FromFamilyName("Microsoft YaHei"),
        };
        canvas.DrawText(text, pos.X, pos.Y, SKTextAlign.Left, font, paint);
    }
    if (is3Channel)
    {
        Cv2.CvtColor(bgraMat, mat, ColorConversionCodes.BGRA2BGR);
    }
    else
    {
        bgraMat.CopyTo(mat);
    }
}