namespace BookRent.User.Api;

public static class UserEndpoints
{
    public static void MapUserEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/user");

        group.MapGet("/userDetails", GetUser);
        group.MapGet("/favourites", GetFavourites);
        group.MapPost("/addFavourite", AddFavourite);
        group.MapDelete("/removeFavourite", RemoveFavourite);
        
        
        
    }
    
    private static IResult GetUser(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    private static IResult GetFavourites(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    private static IResult AddFavourite(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    private static IResult RemoveFavourite(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
}