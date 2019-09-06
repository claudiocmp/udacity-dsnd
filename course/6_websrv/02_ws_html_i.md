## Building a Web Service

### HTML Document Example

Here is an example of HTML code

```
<!DOCTYPE html>

<html>

<head>
    <title>Page Title</title>
</head>

<body>
    <h1>A Photo of a Beautiful Landscape</h1>
    <a href="https://www.w3schools.com/tags">HTML tags</a>
    <p>Here is the photo</p>
    <img src="photo.jpg" alt="Country Landscape">
</body>

</html>
```

### Explanation of the HTML document

As you progress through the lesson, you'll find that the `<head>` tag is mostly for housekeeping like specifying the page title and adding meta tags. Meta tags are in essence information about the page that web crawlers see but users do not. The head tag also contains links to javascript and css files, which you'll see later in the lesson.

The website content goes in the `<body>` tag. The body tag can contain headers, paragraphs, images, links, forms, lists, and a handful of other tags. Of particular note in this example are the link tag `<a>` and the image tag `<img>`.

Both of these tags link to external information outside of the html doc. In the html code above, the link `<a>` tag links to an external website called w3schools. The href is called an attribute, and in this case href specifies the link.

The image `<img>` tag displays an image called "photo.jpg". In this case, the jpg file and the html document are in the same directory, but the documents do not have to be. The src attribute specifies the path to the image file relative to the html document. The alt tag contains text that gets displaced in case the image cannot be found.

Further links: [HTML Reference](https://www.w3schools.com/tags/default.asp)